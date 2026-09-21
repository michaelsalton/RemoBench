#include "remo/PipelineRegistry.h"

#include <algorithm>
#include <cstdio>
#include <string>

#include "remo/PointSource.h"

#include "../../kernels/simlod/simlod_layout.h"

namespace remo {
namespace {

// kernel_construct stops ingesting once the persistent allocator is within this of its
// capacity (progressive_octree_voxels.cu, `safetyMargin`). It is the difference between
// the buffer that was allocated and the buffer that can actually be filled, so a
// prediction that ignores it overstates completion by a few percent on a small budget.
constexpr uint64_t kPersistentSafetyMargin = 200'000'000ull;

}

void PipelineRegistry::add(PipelineFactory factory) {
	if (!factory) return;

	std::unique_ptr<ILodPipeline> probe = factory();
	if (!probe) {
		fprintf(stderr, "remobench: pipeline factory returned null at registration\n");
		return;
	}
	PipelineInfo info = probe->info();
	probe.reset();

	if (info.id.empty()) {
		fprintf(stderr, "remobench: pipeline has an empty id; not registered\n");
		return;
	}
	for (const Entry& e : m_entries) {
		if (e.info.id == info.id) {
			fprintf(stderr, "remobench: duplicate pipeline id '%s'; not registered\n",
			        info.id.c_str());
			return;
		}
	}

	m_infos.push_back(info);
	m_entries.push_back({std::move(info), std::move(factory)});
}

const PipelineInfo* PipelineRegistry::find(const std::string& id) const {
	for (const Entry& e : m_entries) {
		if (e.info.id == id) return &e.info;
	}
	return nullptr;
}

// The input term, which is the whole difference streaming makes: a resident consumer pays
// 16 bytes per point of cloud, a streaming one pays for its ring and nothing more.
static uint64_t inputBytesFor(const PipelineInfo& info, const CloudMeta& meta) {
	if (!info.needsWholeCloudResident && info.ringSlots > 0) {
		// A ring deeper than the cloud is never allocated, so a small cloud is not
		// charged for slots it will not use. See CloudSource::start().
		const uint64_t total = (meta.numPoints + kSlotCapacity - 1) / kSlotCapacity;
		const uint64_t slots = std::min<uint64_t>(info.ringSlots, total);
		return slots * kSlotCapacity * sizeof(Point);
	}
	return meta.numPoints * sizeof(Point);
}

uint64_t PipelineRegistry::minBytesRequired(const PipelineInfo& info,
                                            const CloudMeta& meta) const {
	const double perPoint =
		info.minBytesPerPointEstimate * static_cast<double>(meta.numPoints);
	return static_cast<uint64_t>(perPoint) + inputBytesFor(info, meta) +
	       info.fixedBytesEstimate;
}

uint64_t PipelineRegistry::refusalFloor(const PipelineInfo& info,
                                        const CloudMeta& meta) const {
	// A pipeline that truncates cleanly is refused only when it cannot run at all --
	// when the budget will not hold the ring, the fixed overhead and the smallest store
	// it works with. Everything above that floor is a valid, smaller tree, and how much
	// smaller is completionWarning()'s job to say.
	//
	// This is the distinction the old gate did not make: one constant was both the
	// appetite and the refusal threshold, so a cloud that would have built a perfectly
	// good 61% tree was turned away for not fitting whole.
	if (info.minStoreBytes == 0) return minBytesRequired(info, meta);
	return info.minStoreBytes + inputBytesFor(info, meta) + info.fixedBytesEstimate;
}

bool PipelineRegistry::fits(const PipelineInfo& info, const CloudMeta& meta,
                            const DeviceBudget& budget) const {
	return refusalFloor(info, meta) <= budget.bytes;
}

double PipelineRegistry::predictedCompletion(const PipelineInfo& info,
                                             const CloudMeta& meta,
                                             const DeviceBudget& budget) const {
	if (meta.numPoints == 0) return 1.0;
	// A pipeline with no per-point store of its own either fits whole or is refused;
	// there is no partial outcome to predict.
	if (info.minBytesPerPointEstimate <= 0.0) return fits(info, meta, budget) ? 1.0 : 0.0;

	const uint64_t overhead = inputBytesFor(info, meta) + info.fixedBytesEstimate;
	if (budget.bytes <= overhead) return 0.0;

	const uint64_t room = budget.bytes - overhead;
	const uint64_t want = static_cast<uint64_t>(
		info.bytesPerPointEstimate * static_cast<double>(meta.numPoints));
	const uint64_t persistent = std::min(want, room);
	const uint64_t usable =
		persistent > kPersistentSafetyMargin ? persistent - kPersistentSafetyMargin : 0;

	const double points =
		static_cast<double>(usable) / info.minBytesPerPointEstimate;
	return std::clamp(points / static_cast<double>(meta.numPoints), 0.0, 1.0);
}

std::string PipelineRegistry::completionWarning(const PipelineInfo& info,
                                                const CloudMeta& meta,
                                                const DeviceBudget& budget) const {
	const double fraction = predictedCompletion(info, meta, budget);
	if (fraction >= 0.999) return {};

	char buf[256];
	snprintf(buf, sizeof(buf),
	         "the budget holds about %.0f%% of this cloud; ingest will stop cleanly on a "
	         "truncated tree (memCapacityReached). %.1f GB of budget would hold all of it.",
	         fraction * 100.0,
	         static_cast<double>(minBytesRequired(info, meta)) / 1e9);
	return buf;
}

std::string PipelineRegistry::unsupportedReason(const PipelineInfo& info,
                                                const CloudMeta& meta,
                                                const DeviceBudget& budget) const {
	if (!fits(info, meta, budget)) {
		const double inputGB = static_cast<double>(inputBytesFor(info, meta)) / 1e9;
		char buf[384];
		if (info.minStoreBytes == 0) {
			snprintf(buf, sizeof(buf),
			         "needs %.1f GB for %llu points and cannot build a partial tree "
			         "(%.0f B/pt structure + %.1f GB %s + %.2f GB fixed); "
			         "the budget is %.1f GB",
			         static_cast<double>(minBytesRequired(info, meta)) / 1e9,
			         static_cast<unsigned long long>(meta.numPoints),
			         info.minBytesPerPointEstimate, inputGB,
			         info.needsWholeCloudResident ? "resident cloud" : "ring",
			         static_cast<double>(info.fixedBytesEstimate) / 1e9,
			         static_cast<double>(budget.bytes) / 1e9);
		} else {
			snprintf(buf, sizeof(buf),
			         "cannot run at all below %.1f GB (%.1f GB %s + %.2f GB fixed + "
			         "%.2f GB smallest store); the budget is %.1f GB. The whole cloud "
			         "would need %.1f GB.",
			         static_cast<double>(refusalFloor(info, meta)) / 1e9, inputGB,
			         info.needsWholeCloudResident ? "resident cloud" : "ring",
			         static_cast<double>(info.fixedBytesEstimate) / 1e9,
			         static_cast<double>(info.minStoreBytes) / 1e9,
			         static_cast<double>(budget.bytes) / 1e9,
			         static_cast<double>(minBytesRequired(info, meta)) / 1e9);
		}
		return buf;
	}

	if (meta.isSyntheticFixture && info.id == "cudalod") {
		return "CudaLOD's split kernel faults on the synthetic fixture's point "
		       "distribution (not yet root-caused). Load a real .simlod/.las cloud "
		       "to use this pipeline.";
	}

	// The 50M ceiling, now conditional rather than absolute.
	//
	// It was never a property of SimLOD: kernel_construct reads batch N from slot
	// N % BATCH_STREAM_SIZE, and a flat resident cloud puts batch N at slot N, so the two
	// agree only while N < 50. The ceiling was the shape of RemoBench's feed showing
	// through. PointSource wraps now, so a streaming pipeline has no ceiling at all --
	// and the check stays, addressing the case it was always really about, because a
	// resident feed past 50 batches still silently re-reads slot 0.
	//
	// Deleting it outright was the alternative. It is kept because it is the only thing
	// standing between a wrongly-shaped feed and a tree built from the wrong points with
	// no fault and plausible counts.
	if (info.needsWholeCloudResident && info.ringSlots == 0 && info.progressive &&
	    meta.numPoints > simlod::kMaxAddressablePoints) {
		char buf[384];
		snprintf(buf, sizeof(buf),
		         "%s is configured for a resident cloud, and kernel_construct addresses "
		         "batch N at slot N %% %u, so a resident feed can only describe %.0fM "
		         "points; this cloud has %.0fM. A streaming ring has no such limit.",
		         info.displayName.c_str(), simlod::kBatchStreamSize,
		         static_cast<double>(simlod::kMaxAddressablePoints) / 1e6,
		         static_cast<double>(meta.numPoints) / 1e6);
		return buf;
	}

	return {};
}

bool PipelineRegistry::switchTo(const std::string& id, PointSource* source,
                                const CloudMeta& meta,
                                const DeviceBudget& budget, std::string* err) {
	const Entry* entry = nullptr;
	for (const Entry& e : m_entries) {
		if (e.info.id == id) {
			entry = &e;
			break;
		}
	}
	if (!entry) {
		if (err) *err = "unknown pipeline: " + id;
		return false;
	}

	if (const std::string reason = unsupportedReason(entry->info, meta, budget);
	    !reason.empty()) {
		if (err) *err = entry->info.displayName + ": " + reason;
		return false;
	}

	if (m_active) {
		m_active->release();
		m_active.reset();
		m_activeId.clear();
	}

	// The ingest shape is the incoming pipeline's requirement, so it is decided here and
	// nowhere else -- and before the pipeline allocates, since the two share one budget
	// and the difference between a resident cloud and a ring is most of it. Switching
	// from a stream to a resident pipeline means re-reading from batch 0, which is
	// correct and is what rewind() is for.
	if (source) {
		const PointSource::Mode mode = entry->info.needsWholeCloudResident
		                                   ? PointSource::Mode::Whole
		                                   : PointSource::Mode::Stream;
		if (!source->start(mode, entry->info.ringSlots, budget.bytes, err)) return false;
		source->rewind();
	}

	std::unique_ptr<ILodPipeline> pipeline = entry->factory();
	if (!pipeline) {
		if (err) *err = "factory returned null for " + id;
		return false;
	}

	if (!pipeline->initPrograms(err)) return false;
	if (!pipeline->allocate(meta, budget, err)) return false;
	pipeline->reset();

	m_active = std::move(pipeline);
	m_activeId = id;
	return true;
}

bool PipelineRegistry::reloadForCloud(PointSource* source, const CloudMeta& meta,
                                      const DeviceBudget& budget,
                                      std::string* err) {
	if (!m_active) {
		if (err) *err = "no active pipeline";
		return false;
	}
	m_active->release();

	if (source) {
		const PipelineInfo* info = find(m_activeId);
		const bool resident = !info || info->needsWholeCloudResident;
		const PointSource::Mode mode =
			resident ? PointSource::Mode::Whole : PointSource::Mode::Stream;
		if (!source->start(mode, info ? info->ringSlots : 0, budget.bytes, err)) {
			return false;
		}
		source->rewind();
	}

	if (!m_active->allocate(meta, budget, err)) return false;
	m_active->reset();
	return true;
}

}
