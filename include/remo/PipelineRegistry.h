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

	bool fits(const PipelineInfo& info, const CloudMeta& meta,
	          const DeviceBudget& budget) const;

	std::string unsupportedReason(const PipelineInfo& info, const CloudMeta& meta,
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
