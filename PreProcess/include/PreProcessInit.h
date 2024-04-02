#ifndef __PREPROCESSINIT_H__
#define __PREPROCESSINIT_H__
#include <string>
#include <vector>
#include "rtimages.h"
#include "PreProcessROI.h"

#define DATAMAX 9999.0f
#define DATAMIN -9999.0f


namespace PreProcess{
    bool getROINamesFromJSON(const std::string& json_path,
        std::vector<std::string>& target);
    bool dicomInit(const std::string& dicomFolder,
        RTImage** rtimage, bool verbose=true);
    bool ctdataInitMode1(PreProcess::FloatVolume& ctdata);
    bool densityInitMode1(FloatVolume& density, const PreProcess::FloatVolume& ctdata);
    bool ROIInitModel1(ROIMaskList& roi_list, const FloatVolume& ctdata);
}

#endif