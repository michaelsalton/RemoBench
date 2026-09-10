// GPU timing, owned by the shell rather than by each pipeline.
//
// WHY THIS IS NOT PER-PIPELINE. ILodPipeline.h:5 already states the rule -- a pipeline
// does not own "the window, the camera, the loader, the rasterizer, EDL, the GL blit,
// timing or stats plumbing". Timing was the one item on that list still living inside
// the pipelines, as an ad-hoc CUevent pair per phase, and the consequence was exactly
// what you would predict: SimlodPipeline::render and CudalodPipeline::render simply
// never recorded one, so `renderDeviceMsLast` stayed at its default and the GUI printed
// a plausible-looking 0.00 in the same format as a real measurement. A pipeline could
// FAIL TO MEASURE something and nothing said so.
//
// With the scope at the launch site there is no such failure mode: a launch either sits
// inside a GpuScope or it does not appear in the output at all, and a scope that was
// never recorded has n == 0, which is distinguishable from a measured zero.
//
// TWO REGIMES, NEVER POOLED. Build launches synchronise; render launches only
// synchronise under --strict-timing. A sample read one frame late and a sample read
// after a context sync are not the same measurement, and mixing them silently is how
// two incomparable runs become indistinguishable after the fact. Every sample is
// therefore filed under the Regime that produced it, and the two accumulators are kept
// apart structurally rather than by convention.
//
// COST. Harvesting uses cuEventQuery, never cuEventSynchronize, in the deferred regime:
// a scope whose events are not ready yet is simply left for the next frame's pass. A
// profiler that stalls the pipeline is measuring itself.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda.h>

namespace remo {

// How a sample was read back. Recorded per sample and never pooled across the two --
// see the header comment.
enum class Regime : int {
	Deferred = 0,  // cuEventQuery, read whenever the events happen to be ready
	Strict = 1,    // read after a synchronise, attributed to the frame that produced it
};

const char* regimeName(Regime regime);

// One scope's distribution.
//
// Welford for mean and variance in a single pass with no catastrophic cancellation,
// plus a ring of raw samples for order statistics. A mean alone is close to useless
// here: a progressive builder produces a long tail by construction, and a mean over
// GPU frame times is dominated by it.
class ScopeStats {
public:
	// 4096 x 8 B per scope. Covers a 600-frame bench run several times over, and is
	// small enough that retention needs no configuration.
	static constexpr size_t kRingCapacity = 4096;

	void add(double ms);
	void clear();

	uint64_t count() const { return m_n; }
	bool empty() const { return m_n == 0; }

	double last() const { return m_last; }
	double min() const { return m_n ? m_min : 0.0; }
	double max() const { return m_n ? m_max : 0.0; }
	double mean() const { return m_n ? m_mean : 0.0; }

	// Summed cost over every sample. This is what a throughput figure over a whole
	// progressive build divides by.
	double total() const { return m_mean * static_cast<double>(m_n); }

	double variance() const;
	double stddev() const;

	// Nearest-rank over the retained ring, so for n > kRingCapacity this describes the
	// most recent kRingCapacity samples rather than the whole run. p is 0..1.
	double percentile(double p) const;
	double median() const { return percentile(0.5); }

	// How many samples the ring actually holds, so a reader can tell whether a
	// percentile describes the whole run or only its tail end.
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

	// Sorted copy, rebuilt on demand. The panel asks for median and p95 every frame;
	// re-sorting on every query would be the profiler showing up in its own numbers.
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

	// Harvests whatever completed since the last pass, then opens a frame. `stream` is
	// the stream scopes record on, taken from FrameContext so that a future real stream
	// needs no change here.
	void beginFrame(uint64_t frameIndex, Regime regime, CUstream stream);

	// Closes the frame. In the strict regime this blocks on any scope still outstanding,
	// so its samples are attributed to the frame that produced them; in the deferred
	// regime it is another non-blocking pass.
	void endFrame();

	// --- recording. Prefer GpuScope; these are what it calls. -------------------
	int begin(const char* name);
	void end(int handle);

	// --- reading ---------------------------------------------------------------
	Regime regime() const { return m_regime; }

	// Null when the scope has never been recorded -- which is the state the old
	// three-doubles interface could not express, and the reason a missing measurement
	// used to read as 0.00 ms.
	const ScopeStats* find(const std::string& name) const;
	const ScopeStats* find(const std::string& name, Regime regime) const;

	std::vector<std::string> scopeNames() const;

	// Parent scope name, or empty for a top-level scope. Recorded from the first
	// occurrence, so a flame-style breakdown is available later without re-instrumenting.
	const std::string& parentOf(const std::string& name) const;

	void clear();
	// Drops the stats for every scope whose name starts with `prefix`. Pipelines use it
	// when a rebuild makes previous samples non-comparable -- switching CudaLOD's
	// sampling strategy, for one, where pooling strategy 0 and strategy 3 voxelisation
	// times into one distribution would produce a number describing neither.
	void clearPrefix(const std::string& prefix);

	// Scopes that could not be recorded because the event pool was exhausted. Non-zero
	// means samples are missing, so it is surfaced rather than swallowed.
	uint64_t droppedScopes() const { return m_dropped; }

private:
	struct Scope {
		std::string name;
		std::string parent;
		ScopeStats stats[2];  // indexed by Regime
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

	// A cap, not a target. Events are recycled, so steady state needs only as many
	// pairs as there are scopes outstanding at once -- a handful. The cap exists so a
	// pathological case (events that never complete because the context died) cannot
	// grow the pool without bound.
	static constexpr size_t kMaxEventPairs = 256;

	std::vector<Scope> m_scopes;
	std::unordered_map<std::string, size_t> m_index;

	std::vector<Open> m_open;        // scopes begun but not yet ended, innermost last
	std::vector<Pending> m_pending;  // ended, awaiting readback
	std::vector<std::pair<CUevent, CUevent>> m_freeEvents;

	size_t m_createdPairs = 0;
	uint64_t m_dropped = 0;

	uint64_t m_frameIndex = 0;
	Regime m_regime = Regime::Deferred;
	CUstream m_stream = nullptr;

	std::string m_noParent;
};

// RAII bracket around one launch.
//
//   {
//       GpuScope s(frame.profiler, "simlod.render");
//       REMO_CU(cuLaunchCooperativeKernel(...));
//   }
//
// Takes a POINTER, not the reference the plan sketched: FrameContext::profiler is
// documented as never null in a normal frame, but a null dereference inside a render
// loop is a worse outcome than a silently unmeasured scope, and a headless path that
// forgets to set it should not crash.
//
// Keep the scope tight around the launch. A scope that spans an early return brackets
// no GPU work and contributes a ~0 ms sample, which pollutes the distribution it exists
// to describe.
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

}  // namespace remo
