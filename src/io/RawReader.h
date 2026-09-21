#pragma once

#include <string>
#include <vector>

#include "remo/PointSource.h"

namespace remo {

bool readSimlod(const std::string& path, CloudMeta& meta,
                std::vector<Point>& points, std::string* err);

}
