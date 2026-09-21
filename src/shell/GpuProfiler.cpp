#include "remo/GpuProfiler.h"

#include <algorithm>
#include <cmath>

#include "remo/CudaCheck.h"

namespace remo {

const char* regimeName(Regime regime) {
	return regime == Regime::Strict ? "strict" : "deferred";
}

void ScopeStats::add(double ms) {
	++m_n;
	m_last = ms;
	if (m_n == 1) {
		m_min = m_max = ms;
	} else {
		m_min = std::min(m_min, ms);
		m_max = std::max(m_max, ms);
	}

	const double delta = ms - m_mean;
	m_mean += delta / static_cast<double>(m_n);
	m_m2 += delta * (ms - m_mean);

	if (m_ring.size() < kRingCapacity) {
		m_ring.push_back(ms);
	} else {
		m_ring[m_ringHead] = ms;
		m_ringHead = (m_ringHead + 1) % kRingCapacity;
	}
	m_sortedDirty = true;
}

void ScopeStats::clear() {
	m_n = 0;
	m_min = m_max = m_mean = m_m2 = m_last = 0.0;
	m_ring.clear();
	m_ringHead = 0;
	m_sorted.clear();
	m_sortedDirty = true;
}

double ScopeStats::variance() const {
	if (m_n < 2) return 0.0;
	return m_m2 / static_cast<double>(m_n - 1);
}

double ScopeStats::stddev() const { return std::sqrt(variance()); }

double ScopeStats::percentile(double p) const {
	if (m_ring.empty()) return 0.0;
	if (m_sortedDirty) {
		m_sorted = m_ring;
		std::sort(m_sorted.begin(), m_sorted.end());
		m_sortedDirty = false;
	}
	const double clamped = std::clamp(p, 0.0, 1.0);
	size_t rank = static_cast<size_t>(std::ceil(clamped * static_cast<double>(m_sorted.size())));
	if (rank == 0) rank = 1;
	if (rank > m_sorted.size()) rank = m_sorted.size();
	return m_sorted[rank - 1];
}

GpuProfiler::~GpuProfiler() { destroyEvents(); }

void GpuProfiler::destroyEvents() {
	for (const Open& o : m_open) {
		if (o.start) cuEventDestroy(o.start);
		if (o.end) cuEventDestroy(o.end);
	}
	m_open.clear();

	for (const Pending& p : m_pending) {
		if (p.start) cuEventDestroy(p.start);
		if (p.end) cuEventDestroy(p.end);
	}
	m_pending.clear();

	for (const auto& pair : m_freeEvents) {
		if (pair.first) cuEventDestroy(pair.first);
		if (pair.second) cuEventDestroy(pair.second);
	}
	m_freeEvents.clear();
	m_createdPairs = 0;
}

size_t GpuProfiler::scopeIndex(const char* name) {
	std::string key(name);
	auto it = m_index.find(key);
	if (it != m_index.end()) return it->second;

	const size_t idx = m_scopes.size();
	Scope scope;
	scope.name = key;
	m_scopes.push_back(std::move(scope));
	m_index.emplace(std::move(key), idx);
	return idx;
}

bool GpuProfiler::acquireEvents(CUevent* start, CUevent* end) {
	if (!m_freeEvents.empty()) {
		*start = m_freeEvents.back().first;
		*end = m_freeEvents.back().second;
		m_freeEvents.pop_back();
		return true;
	}
	if (m_createdPairs >= kMaxEventPairs) return false;

	CUevent a = nullptr;
	CUevent b = nullptr;
	if (cuEventCreate(&a, CU_EVENT_DEFAULT) != CUDA_SUCCESS) return false;
	if (cuEventCreate(&b, CU_EVENT_DEFAULT) != CUDA_SUCCESS) {
		cuEventDestroy(a);
		return false;
	}
	++m_createdPairs;
	*start = a;
	*end = b;
	return true;
}

void GpuProfiler::recycle(CUevent start, CUevent end) {
	m_freeEvents.emplace_back(start, end);
}

void GpuProfiler::beginFrame(uint64_t frameIndex, Regime regime, CUstream stream) {
	m_frameIndex = frameIndex;
	m_regime = regime;
	m_stream = stream;

	harvest(false);
}

void GpuProfiler::endFrame() {
	for (const Open& o : m_open) {
		recycle(o.start, o.end);
		++m_dropped;
	}
	m_open.clear();

	harvest(m_regime == Regime::Strict);
}

int GpuProfiler::begin(const char* name) {
	const size_t idx = scopeIndex(name);

	if (!m_open.empty()) {
		const std::string& parent = m_scopes[m_open.back().scope].name;
		if (m_scopes[idx].parent.empty() && parent != m_scopes[idx].name) {
			m_scopes[idx].parent = parent;
		}
	}

	Open open;
	open.scope = idx;
	if (!acquireEvents(&open.start, &open.end)) {
		++m_dropped;
		open.start = open.end = nullptr;
		m_open.push_back(open);
		return static_cast<int>(m_open.size()) - 1;
	}

	if (cuEventRecord(open.start, m_stream) != CUDA_SUCCESS) {
		recycle(open.start, open.end);
		++m_dropped;
		open.start = open.end = nullptr;
	}
	m_open.push_back(open);
	return static_cast<int>(m_open.size()) - 1;
}

void GpuProfiler::end(int handle) {
	if (handle < 0 || m_open.empty()) return;
	if (handle != static_cast<int>(m_open.size()) - 1) return;

	const Open open = m_open.back();
	m_open.pop_back();

	if (!open.start || !open.end) return;

	if (cuEventRecord(open.end, m_stream) != CUDA_SUCCESS) {
		recycle(open.start, open.end);
		++m_dropped;
		return;
	}

	Pending pending;
	pending.scope = open.scope;
	pending.start = open.start;
	pending.end = open.end;
	pending.regime = m_regime;
	m_pending.push_back(pending);
}

void GpuProfiler::addSample(const char* name, double ms) {
	if (!name || !(ms >= 0.0)) return;
	m_scopes[scopeIndex(name)].stats[static_cast<int>(m_regime)].add(ms);
}

void GpuProfiler::harvest(bool blocking) {
	size_t keep = 0;
	for (size_t i = 0; i < m_pending.size(); ++i) {
		Pending& p = m_pending[i];

		const CUresult status =
			blocking ? cuEventSynchronize(p.end) : cuEventQuery(p.end);

		if (status == CUDA_ERROR_NOT_READY) {
			m_pending[keep++] = p;
			continue;
		}

		if (status == CUDA_SUCCESS) {
			float ms = 0.0f;
			if (cuEventElapsedTime(&ms, p.start, p.end) == CUDA_SUCCESS) {
				m_scopes[p.scope].stats[static_cast<int>(p.regime)].add(
					static_cast<double>(ms));
			} else {
				++m_dropped;
			}
		} else {
			++m_dropped;
		}

		recycle(p.start, p.end);
	}
	m_pending.resize(keep);
}

const ScopeStats* GpuProfiler::find(const std::string& name) const {
	return find(name, m_regime);
}

const ScopeStats* GpuProfiler::find(const std::string& name, Regime regime) const {
	auto it = m_index.find(name);
	if (it == m_index.end()) return nullptr;
	const ScopeStats& stats = m_scopes[it->second].stats[static_cast<int>(regime)];
	if (stats.empty()) return nullptr;
	return &stats;
}

std::vector<std::string> GpuProfiler::scopeNames() const {
	std::vector<std::string> names;
	names.reserve(m_scopes.size());
	for (const Scope& s : m_scopes) names.push_back(s.name);
	std::sort(names.begin(), names.end());
	return names;
}

const std::string& GpuProfiler::parentOf(const std::string& name) const {
	auto it = m_index.find(name);
	if (it == m_index.end()) return m_noParent;
	return m_scopes[it->second].parent;
}

void GpuProfiler::clear() {
	for (Scope& s : m_scopes) {
		s.stats[0].clear();
		s.stats[1].clear();
	}
	m_dropped = 0;
}

void GpuProfiler::clearPrefix(const std::string& prefix) {
	for (Scope& s : m_scopes) {
		if (s.name.rfind(prefix, 0) != 0) continue;
		s.stats[0].clear();
		s.stats[1].clear();
	}
}

}
