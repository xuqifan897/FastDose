#ifndef __PREPROCESSINIT_H__
#define __PREPROCESSINIT_H__
#include <string>
#include <vector>
#include "rtimages.h"

namespace PreProcess{
    bool getROINamesFromJSON(const std::string& json_path,
        std::vector<std::string>& target);
    bool dicomInit(const std::string& dicomFolder,
        RTImage** rtimage, bool verbose=true);
}

#endif