#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda.h>

namespace remo {

enum class Regime : int {
	Deferred = 0,
	Strict = 1,
};

const char* regimeName(Regime regime);

class ScopeStats {
public:
	static constexpr size_t kRingCapacity = 4096;

	void add(double ms);
	void clear();

	uint64_t count() const { return m_n; }
	bool empty() const { return m_n == 0; }

	double last() const { return m_last; }
	double min() const { return m_n ? m_min : 0.0; }
	double max() const { return m_n ? m_max : 0.0; }
	double mean() const { return m_n ? m_mean : 0.0; }

	double total() const { return m_mean * static_cast<double>(m_n); }

	double variance() const;
	double stddev() const;

	double percentile(double p) const;
	double median() const { return percentile(0.5); }

	size_t retained() const { return m_ring.size(); }

private:
	uint64_t m_n = 0;
	double m_min = 0.0;
	double m_max = 0.0;
	double m_mean = 0.0;
	double m_m2 = 0.0;
	double m_last = 0.0;

	std::vector<double> m_ring;
	size_t m_ringHead = 0;

	mutable std::vector<double> m_sorted;
	mutable bool m_sortedDirty = true;
};

class GpuScope;

class GpuProfiler {
public:
	GpuProfiler() = default;
	~GpuProfiler();

	GpuProfiler(const GpuProfiler&) = delete;
	GpuProfiler& operator=(const GpuProfiler&) = delete;

	void beginFrame(uint64_t frameIndex, Regime regime, CUstream stream);

	void endFrame();

	int begin(const char* name);
	void end(int handle);

	// Record a sample measured by something other than a CUevent pair, into the
	// current regime. This is how intra-kernel phases reach the profiler:
	// cuEventRecord is stream-ordered and cannot bracket a phase inside a single
	// cooperative launch, so those durations are timed on the device and handed
	// over here. See plans/02_ProfilingTools.md Layer 3.
	void addSample(const char* name, double ms);

	Regime regime() const { return m_regime; }

	const ScopeStats* find(const std::string& name) const;
	const ScopeStats* find(const std::string& name, Regime regime) const;

	std::vector<std::string> scopeNames() const;

	const std::string& parentOf(const std::string& name) const;

	void clear();
	void clearPrefix(const std::string& prefix);

	uint64_t droppedScopes() const { return m_dropped; }

private:
	struct Scope {
		std::string name;
		std::string parent;
		ScopeStats stats[2];
	};

	struct Open {
		size_t scope = 0;
		CUevent start = nullptr;
		CUevent end = nullptr;
	};

	struct Pending {
		size_t scope = 0;
		CUevent start = nullptr;
		CUevent end = nullptr;
		Regime regime = Regime::Deferred;
	};

	size_t scopeIndex(const char* name);
	bool acquireEvents(CUevent* start, CUevent* end);
	void recycle(CUevent start, CUevent end);
	void harvest(bool blocking);
	void destroyEvents();

	static constexpr size_t kMaxEventPairs = 256;

	std::vector<Scope> m_scopes;
	std::unordered_map<std::string, size_t> m_index;

	std::vector<Open> m_open;
	std::vector<Pending> m_pending;
	std::vector<std::pair<CUevent, CUevent>> m_freeEvents;

	size_t m_createdPairs = 0;
	uint64_t m_dropped = 0;

	uint64_t m_frameIndex = 0;
	Regime m_regime = Regime::Deferred;
	CUstream m_stream = nullptr;

	std::string m_noParent;
};

class GpuScope {
public:
	GpuScope(GpuProfiler* profiler, const char* name)
		: m_profiler(profiler), m_handle(profiler ? profiler->begin(name) : -1) {}

	~GpuScope() {
		if (m_profiler) m_profiler->end(m_handle);
	}

	GpuScope(const GpuScope&) = delete;
	GpuScope& operator=(const GpuScope&) = delete;

private:
	GpuProfiler* m_profiler;
	int m_handle;
};

}
