// Pipeline registry and runtime switching.
//
// Switching is deliberately EXCLUSIVE by default: the outgoing pipeline releases
// its device memory before the incoming one allocates. Two reasons, and the second
// is the important one:
//
//   - There is no way to hand SimLOD's chunked octree to CudaLOD's
//     contiguous-slice traversal, so "switch and keep the structure" is not a thing
//     that exists. A switch always means rebuild.
//   - Both pipelines must be offered the SAME byte budget, or their memory figures
//     are not comparable. Co-residency halves it.
//
// Co-residency is therefore opt-in (instant toggling on a small cloud is genuinely
// useful when eyeballing differences) and must be off whenever a number is being
// recorded.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "remo/ILodPipeline.h"

namespace remo {

class PointSource;

class PipelineRegistry {
public:
	// Register a pipeline. Its PipelineInfo is read from a throwaway instance, so
	// ILodPipeline::info() is the SINGLE source of truth for a pipeline's id, display
	// name and memory estimate.
	//
	// The alternative -- passing a PipelineInfo alongside the factory -- means every
	// pipeline spells those fields out twice, and they can disagree. That matters
	// beyond tidiness: the capability gate that greys out a pipeline reads
	// bytesPerPointEstimate from the registry's copy, so a mismatch would have the
	// GUI gating on one number while the pipeline allocates against another.
	//
	// Constructing an instance here is cheap by contract: ILodPipeline constructors
	// do no work, because compilation lives in initPrograms() and allocation in
	// allocate().
	void add(PipelineFactory factory);

	const std::vector<PipelineInfo>& list() const { return m_infos; }

	ILodPipeline* active() const { return m_active.get(); }
	const std::string& activeId() const { return m_activeId; }

	// Can this pipeline handle this cloud within the budget? On a 16GB card this is
	// not academic -- CudaLOD needs ~44 bytes/point all told, so a 350M cloud is
	// ~15.4GB and does not fit, while SimLOD streams the same file. The GUI greys
	// the entry out and the benchmark harness records "skipped_capacity" rather than
	// letting it crash.
	bool fits(const PipelineInfo& info, const CloudMeta& meta,
	          const DeviceBudget& budget) const;

	// Empty if this pipeline can handle this cloud; otherwise a human-readable reason,
	// shown as a tooltip on the disabled entry.
	//
	// Covers more than memory. A pipeline that faults on a given input must be refused
	// up front, because a device fault kills the CUDA context outright -- there is no
	// recovering and continuing, so it would take the whole session down from a single
	// click.
	std::string unsupportedReason(const PipelineInfo& info, const CloudMeta& meta,
	                              const DeviceBudget& budget) const;

	// release(old) -> allocate(new, same budget) -> reset() -> rewind(source).
	bool switchTo(const std::string& id, PointSource* source,
	              const CloudMeta& meta, const DeviceBudget& budget,
	              std::string* err);

	// Re-run allocate/reset for the active pipeline against a new cloud.
	bool reloadForCloud(PointSource* source, const CloudMeta& meta,
	                    const DeviceBudget& budget, std::string* err);

private:
	struct Entry {
		PipelineInfo info;
		PipelineFactory factory;
	};

	std::vector<Entry> m_entries;
	std::vector<PipelineInfo> m_infos;

	std::unique_ptr<ILodPipeline> m_active;
	std::string m_activeId;
};

}  // namespace remo
