#ifndef __PREPROCESSROI_H__
#define __PREPROCESSROI_H__

#include <string>
#include <vector>
#include <list>
#include "cuda_runtime.h"
#include "PreProcessHelper.h"
#include "rtstruct.h"

namespace PreProcess {
    class BaseROIMask {
    public:
        BaseROIMask() {}
        BaseROIMask(std::string name, ArrayProps props) : name{name}, props(props) {}
        virtual ~BaseROIMask() = 0;

        std::string name;       // name of ROI structure
        ArrayProps props;       // custom bbox containing ROI for more efficient iterating

        virtual uint8_t operator[](uint64_t idx) = 0;
        uint nvoxels() { return props.nvoxels(); }
        virtual uint nones() = 0;
        uint nzeros() { return nvoxels() - nones(); }

        int writeToFile(std::string fname, bool verbose=false);
        virtual int _writeToHDF5(H5::Group& h5group) = 0;
    };

    class DenseROIMask : public BaseROIMask {
        public:
            DenseROIMask() {}
            DenseROIMask(std::string name, std::vector<uint8_t> mask, ArrayProps props) : BaseROIMask(name, props), mask{mask} {}
            virtual ~DenseROIMask() {}
            std::vector<uint8_t> mask; // linearized binary array (1: in ROI, 0: out ROI)
            virtual uint8_t operator[](uint64_t idx);
            virtual uint nones();

            int _writeToHDF5(H5::Group& h5group);
            static int _readFromHDF5(DenseROIMask& mask, H5::Group& h5group);

        protected:
            uint _nones_cached = 0;
    };


    class ROIMaskList {
    public:
        ROIMaskList() {}
        ~ROIMaskList() {}
        std::vector<std::shared_ptr<DenseROIMask> > _coll;
        void push_back(DenseROIMask* mask) { _coll.emplace_back(mask); }
        std::vector<std::string> getROINames();
        std::vector<uint64_t>    getROICapacities();

        uint64_t size() { return _coll.size(); }

        int writeToFile(std::string fname, bool verbose=false);
        static int readFromFile(ROIMaskList& masklist, std::string fname, bool verbose=false);
        int _writeToHDF5(H5::Group& h5group);
        static int _readFromHDF5(ROIMaskList& masklist, H5::Group& h5group, bool verbose=false);
    };

    bool ROI_init(ROIMaskList& roi_list, const std::vector<std::string>& roi_names,
        RTStruct& rtstruct, const FrameOfReference& frameofref,
        const FloatVolume& ctdata, const FloatVolume& density, bool verbose);

    Volume<uint8_t> generateContourMask(StructureSet&, FrameOfReference,
        FrameOfReference, void* texRay=NULL);
}

#endif