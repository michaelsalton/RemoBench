#pragma once

#include <memory>
#include <string>
#include <vector>

#include "remo/ILodPipeline.h"

namespace remo {

class PointSource;

class PipelineRegistry {
public:
	void add(PipelineFactory factory);

	const std::vector<PipelineInfo>& list() const { return m_infos; }

	ILodPipeline* active() const { return m_active.get(); }
	const std::string& activeId() const { return m_activeId; }

	const PipelineInfo* find(const std::string& id) const;

	// What the pipeline needs on the device before ingest stops: the per-point floor,
	// the input term (whole cloud, or a fixed-size ring), and the fixed overhead.
	uint64_t minBytesRequired(const PipelineInfo& info, const CloudMeta& meta) const;

	// The budget below which the pipeline cannot run at all, as opposed to running and
	// producing a smaller tree. This, not minBytesRequired(), is what fits() refuses on.
	uint64_t refusalFloor(const PipelineInfo& info, const CloudMeta& meta) const;

	bool fits(const PipelineInfo& info, const CloudMeta& meta,
	          const DeviceBudget& budget) const;

	std::string unsupportedReason(const PipelineInfo& info, const CloudMeta& meta,
	                              const DeviceBudget& budget) const;

	// Fraction of the cloud the budget is predicted to hold, in [0, 1]. A cloud that
	// clears fits() can still be truncated by memCapacityReached, and this says by how
	// much before the run starts. The dump reports the observed fraction beside it, so
	// the prediction is checkable rather than decorative.
	double predictedCompletion(const PipelineInfo& info, const CloudMeta& meta,
	                           const DeviceBudget& budget) const;

	// Empty when the whole cloud is predicted to fit; otherwise one line saying so.
	std::string completionWarning(const PipelineInfo& info, const CloudMeta& meta,
	                              const DeviceBudget& budget) const;

	bool switchTo(const std::string& id, PointSource* source,
	              const CloudMeta& meta, const DeviceBudget& budget,
	              std::string* err);

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

}
