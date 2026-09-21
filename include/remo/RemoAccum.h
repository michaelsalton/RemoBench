#pragma once

namespace remo {

enum GeomSum : unsigned int {
	kSumX = 0, kSumY, kSumZ,
	kSumXX, kSumXY, kSumXZ,
	kSumYY, kSumYZ, kSumZZ,
	kNumGeomSums
};

enum ColorSum : unsigned int {
	kSumR = 0, kSumG, kSumB,
	kSumRR, kSumGG, kSumBB,
	kNumColorSums
};

enum AccumState : unsigned int {
	kAccumOpen      = 0,
	kAccumClosed    = 1,
	kAccumFinalized = 2,
};

struct NodeAccum {
	double       g[kNumGeomSums];
	float        c[kNumColorSums];
	unsigned int count;
	unsigned int lastTouchedBatch;
	unsigned int state;
	float        score;
};

struct AccumGlobals {
	unsigned long long mortonWatermark;
	unsigned long long numAccumulated;
	unsigned long long sumLeafCounts;
	unsigned int       pointsFolded;
	unsigned int       numLeavesFolded;
	unsigned int       innerWithSums;
	unsigned int       pad0;
};

struct AccumArgs {
	void*        nodes;
	void*        accums;
	void*        globals;
	unsigned int numNodes;
	unsigned int batchIndex;
	float        octreeMinX;
	float        octreeMinY;
	float        octreeMinZ;
	float        octreeSize;
};

static_assert(sizeof(NodeAccum) == 112, "NodeAccum layout changed; see plans/03_AccumulatorHook.md");
static_assert(sizeof(AccumGlobals) == 40, "AccumGlobals layout changed");
static_assert(sizeof(AccumArgs) == 48, "AccumArgs layout changed; host and device must agree");

}
