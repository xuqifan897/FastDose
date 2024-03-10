#include "PreProcessROI.h"
#include "make_tex_surf.cuh"
#include <helper_cuda.h>

#if WITH_OPENCV2
#include <opencv2/core/core.hpp>
#else
#include <opencv2/imgproc.hpp>
#endif

#include "PreProcessArgs.h"
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

PreProcess::BaseROIMask::~BaseROIMask() {}
int PreProcess::BaseROIMask::writeToFile(std::string fname, bool verbose) {
    // open hdf5 file
    H5::H5File h5file = H5::H5File(fname, H5F_ACC_TRUNC);
    H5::Group rootgroup = h5file.openGroup("/");
    if (!_writeToHDF5(rootgroup)) {
        if (verbose){ std::cout << "Failed to write ROIMask to \""<<fname<<"\""<<std::endl; }
        return false;
    }
    return true;
}

uint8_t PreProcess::DenseROIMask::operator[](uint64_t idx) {
    if (idx >= nvoxels()) { return false; }
    return mask[idx];
}

uint PreProcess::DenseROIMask::nones() {
    // cache if not done yet, otherwise get from cache
    if (_nones_cached == 0) {
        uint n = 0;
        for (uint i=0; i<nvoxels(); ++i) { if (mask[i] == 1) { ++n; } }
        _nones_cached = n;
    }
    return _nones_cached;
}

// write/read HDF5
int PreProcess::DenseROIMask::_writeToHDF5(H5::Group& h5group) {
    H5::DataSpace scalarspace;

    // create attribute - name
    {
        H5::StrType str_t{H5::PredType::C_S1, name.length()+1};
        auto att = h5group.createAttribute("name", str_t, scalarspace);
        att.write(str_t, name);
    }

    // write array props to group
    auto array_props_group = h5group.createGroup("ArrayProps");
    props._writeToHDF5(array_props_group);

    // create dataset - dense mask
    if (nvoxels() > 0) {
        hsize_t dims[] = { nvoxels() };
        H5::DataSpace simplespace(1, dims);
        // no bool in HDF5, use int 0,1 instead
        auto dset = h5group.createDataSet("mask", H5::PredType::STD_U8LE, simplespace);
        dset.write(mask.data(), H5::PredType::NATIVE_UINT8);
    }

    return true;
}

int PreProcess::DenseROIMask::_readFromHDF5(DenseROIMask& mask, H5::Group& h5group) {
    // read attribute - name
    {
        auto att = h5group.openAttribute("name");
        H5::DataType str_t = att.getDataType();
        H5std_string buf("");
        att.read(str_t, buf);
        mask.name = buf;
    }

    // read group - ArrayProps
    {
        H5::Group props_group = h5group.openGroup("ArrayProps");
        ArrayProps props {};
        ArrayProps::_readFromHDF5(props, props_group);
        mask.props = props;
    }

    // read dataset - dense mask
    {
        H5::DataSet dset = h5group.openDataSet("mask");
        H5::DataSpace dspace = dset.getSpace();
        hsize_t N;
        dspace.getSimpleExtentDims(&N, NULL);
        mask.mask = std::vector<uint8_t>(N);
        dset.read(mask.mask.data(), H5::PredType::NATIVE_UINT8, dspace, dspace);
    }

    return true;
}


int PreProcess::ROIMaskList::writeToFile(std::string fname, bool verbose) {
    // open hdf5 file
    H5::H5File h5file = H5::H5File(fname, H5F_ACC_TRUNC);
    H5::Group rootgroup = h5file.openGroup("/");
    if (!_writeToHDF5(rootgroup)) {
        if (verbose){ std::cout << "Failed to write ROIMaskList to \""<<fname<<"\""<<std::endl; }
        return false;
    }
    return true;
}


int PreProcess::ROIMaskList::readFromFile(ROIMaskList& masklist, std::string fname, bool verbose) {
    H5::Exception::dontPrint();
    try {
        masklist = ROIMaskList();
        H5::H5File h5file = H5::H5File(fname, H5F_ACC_RDONLY);
        H5::Group rootgroup = h5file.openGroup("/");
        if (!PreProcess::ROIMaskList::_readFromHDF5(masklist, rootgroup)) {
            if (verbose){ std::cout << "Failed to read ROIMaskList from \""<<fname<<"\""<<std::endl; }
            return false;
        }
    }
    catch (H5::FileIException &file_exists_error) {
        if (verbose){ std::cout << "Failed to read ROIMaskList from \""<<fname<<"\""<<std::endl; }
        return false;
    }
    return true;
}


int PreProcess::ROIMaskList::_writeToHDF5(H5::Group& h5group) {
    H5::DataSpace scalarspace;

    // store each ROIMask to its own group
    uint index = 0;
    for (auto&& roi : _coll) {
        ++index;
        auto roi_group = h5group.createGroup(roi->name);
        if (!roi->_writeToHDF5(roi_group)) { return false; }

        // also store 1-based index for each indicating order
        H5::Attribute att = roi_group.createAttribute("index", H5::PredType::STD_U16LE, scalarspace);
        att.write(H5::PredType::NATIVE_UINT, &index);
    }
    return true;
}


int PreProcess::ROIMaskList::_readFromHDF5(ROIMaskList& masklist, H5::Group& h5group, bool verbose) {
    // read groupnames and indices
    struct IndexedString { uint16_t idx; std::string str; };
    struct OpData { OpData(H5::Group& g) : h5group{g} {}; std::list<IndexedString> groups={}; H5::Group& h5group; };
    OpData opdata(h5group);
    int iter_idx = 0; // iter_count is returned here
    h5group.iterateElems(".", &iter_idx,
        [](hid_t loc_id, const char* name, void* opdata) -> herr_t {
            // iterator body
            // construct an IndexedString for each group and add to "groups" list
            OpData* data = static_cast<OpData*>(opdata);
            H5::Group roi_group = data->h5group.openGroup(name);
            auto att = roi_group.openAttribute("index");
            uint16_t index; att.read(H5::PredType::NATIVE_UINT16, &index);
            data->groups.push_back( IndexedString{index, std::string(name)} );
            return 0;
        }, (void*)&opdata);

    // sort (index, groupname) list on index ascending
    opdata.groups.sort( [](IndexedString& a, IndexedString& b) -> bool {
            // true if a belongs before b
            return (a.idx <= b.idx);
            } );

    if (verbose) { std::cout << "Reading ROIMaskList ("<<iter_idx<<"):" << std::endl; }
    for (auto v : opdata.groups) {
        if (verbose) { std::cout << v.idx << ": " << v.str << std::endl; }
        // open group and load ROI data
        H5::Group roi_group = h5group.openGroup(v.str);
        std::shared_ptr<DenseROIMask> mask = std::make_shared<DenseROIMask>();
        if (!DenseROIMask::_readFromHDF5(*mask, roi_group)) {
            if (verbose){ std::cout << "Failed to read ROIMaskList from H5 Group: \""<<v.str<<"\""<<std::endl; }
            return false;
        }
        masklist._coll.emplace_back(mask);
    }
    return true;
}

bool PreProcess::ROI_init(
    ROIMaskList& roi_list, const std::vector<std::string>& roi_names,
    RTStruct& rtstruct, const FrameOfReference& frameofref,
    const FloatVolume& ctdata, const FloatVolume& density, bool verbose
) {
    for (const auto& roi_name : roi_names) {
        // Check up for duplicates
        bool unique = true;
        for (const auto& r : roi_list.getROINames()) {
            if (r == roi_name) {
                unique = false;
                break;
            }
        }
        if (! unique) {
            std::cout << "Excluding redundant ROI specifications: " << roi_name << std::endl;
            continue;
        }

        // validate name against rtstruct
        int roi_idx = getROIIndex(rtstruct, roi_name, true, verbose);
        if (roi_idx < 0) {
            std::cout << "No contour could be matched from search string: " << roi_name
                << ". skipping" << std::endl;
            continue;
        } else {
            printf("Structure found: #%d - %s\n",roi_idx+1, roi_name.c_str());
        }
        StructureSet roi;
        if (loadStructureSet(roi, rtstruct, roi_idx, verbose)) {
            std::cout << "Failed to load ROI Data for: \""<< roi_name <<"\"" << std::endl;
            return 1;
        }

        // Construct BaseROIMask using rtstruct contour data from file
        ArrayProps roi_bbox = getROIExtents(roi, frameofref, verbose);
        Volume<uint8_t> roi_mask {};
        std::cout << "Creating ROI Mask" << std::endl;
        roi_mask = generateContourMask(roi, ctdata.get_frame(), density.get_frame());

        // crop mask to roi_bbox
        std::vector<uint8_t> cropped_mask(roi_bbox.nvoxels());
        for (uint ii=0; ii<roi_bbox.crop_size.x; ii++) {
            for (uint jj=0; jj<roi_bbox.crop_size.y; jj++) {
                for (uint kk=0; kk<roi_bbox.crop_size.z; kk++) {
                    uint64_t full_key = ((kk+roi_bbox.crop_start.z)*roi_bbox.size.y + (jj+roi_bbox.crop_start.y))*roi_bbox.size.x + (ii+roi_bbox.crop_start.x); // iterator over roi_bbox volume
                    uint64_t crop_key = (kk*roi_bbox.crop_size.y + jj)*roi_bbox.crop_size.x + ii;
                    cropped_mask.at(crop_key) = roi_mask.at(full_key);
                }
            }
        }
        roi_list.push_back(new DenseROIMask(roi_name, cropped_mask, roi_bbox));
        std::cout << std::endl;
    }
    return 0;
}



PreProcess::Volume<uint8_t> PreProcess::generateContourMask(StructureSet& contour,
    FrameOfReference ctframe, FrameOfReference densframe, void* texRay ) {
  // iterate over axial slices, converting contour point coords list to closed polygon
  // Construct volume in original dicom coordsys from rasterized slices
  // resample dicom coordsys to dose coordsys
  Volume<uint8_t> mask = Volume<uint8_t>(ctframe);

  // for each slice in contour point set, determine where in total volume it exists
  unsigned long mark = 0;
  for (int c=0; c<contour.sub_cntr_count; c++) {
    float cntr_z = 0.1f*contour.points[3*mark+2];
    int z_offset = round((cntr_z - mask.start.z) / mask.voxsize.z);

    // use opencv to draw polygon
    // make sure to += to existing slice data to handle multiple contours in same slice
    // binary thresholding will be done later
    cv::Mat mat = cv::Mat::zeros(mask.size.y, mask.size.x, CV_8UC1);
    cv::Point* pts = new cv::Point[contour.sub_cntr_points_count[c]];

    // generate point set and fill polygon
    for (int p=0; p<contour.sub_cntr_points_count[c]; p++) {
      int idx_x = floor((0.1f*contour.points[3*(mark+p)    ] - mask.start.x) / mask.voxsize.x);
      int idx_y = floor((0.1f*contour.points[3*(mark+p) + 1] - mask.start.y) / mask.voxsize.y);
      pts[p] = cv::Point{idx_x, idx_y};
    }

    // render polygon
    cv::Point* pts_array[1] = { pts };
    const int npts[1] = { (int)contour.sub_cntr_points_count[c] };
    cv::fillPoly(mat, (const cv::Point**)pts_array, npts, 1, cv::Scalar(1));
    delete[] pts;

    // copy polygon to mask volume
    long int memoffset = sizeof(uint8_t)*mask.size.x*mask.size.y*z_offset;
    for (int r=0; r<mat.rows; r++) {
      for (int c=0; c<mat.cols; c++) {
        int ptr = memoffset+r*mat.cols+c;
        mask._vect[ptr] = (uint8_t)((mat.at<uint8_t>(r, c)+mask._vect[ptr])>0);
      }
    }
    mark += contour.sub_cntr_points_count[c];
  }


  // bind the mask to a texture for resampling
  /////////////////// Bind Inputs to 3D Texture Arrays //////////////////////////////////////////////
  cudaArray* arrayMask;
  cudaTextureObject_t texMask;
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uint8_t>();
  cudaExtent extent  = make_cudaExtent(mask.size.x, mask.size.y, mask.size.z);
  checkCudaErrors( cudaMalloc3DArray(&arrayMask, &channelDesc, extent) );

  cudaMemcpy3DParms parms = {0};
  parms.srcPtr   = make_cudaPitchedPtr((void*)mask.data(), extent.width*sizeof(uint8_t), extent.width, extent.height);
  parms.dstArray = arrayMask;
  parms.extent   = extent;
  parms.kind     = cudaMemcpyHostToDevice;

  cudaMemcpy3D(&parms);
  makeTexObject<cudaArray>(&texMask, arrayMask, 3, cudaAddressModeBorder, cudaFilterModePoint);
  //////////////////////////////////////////////////////////////////////////////////////////////

  uint8_t* d_remask;
  checkCudaErrors( cudaMalloc(&d_remask, sizeof(uint8_t)*densframe.nvoxels()) );
  dim3 rsBlock = dim3{16,16,1};
  dim3 rsGrid = dim3{
    (uint)ceilf(densframe.size.x/rsBlock.x),
    (uint)ceilf(densframe.size.y/rsBlock.y),
    (uint)ceilf(densframe.size.z/rsBlock.z),
  };
  cudaResample<<<rsGrid, rsBlock>>>(
      d_remask, densframe.start, densframe.size, densframe.spacing,
      texMask, ctframe.start,    ctframe.size,   ctframe.spacing
      );
  cudaThreshold<<<rsGrid, rsBlock>>>(
      d_remask, d_remask, densframe.size, (uint8_t)1
      );

  Volume<uint8_t> remask(densframe);
  checkCudaErrors( cudaMemcpy((void*)remask.data(), d_remask, sizeof(uint8_t)*densframe.nvoxels(), cudaMemcpyDeviceToHost) );
  checkCudaErrors( cudaFree(d_remask) );
  checkCudaErrors( cudaDestroyTextureObject(texMask) );
  checkCudaErrors( cudaFreeArray(arrayMask) );

  return remask;
}


std::vector<std::string> PreProcess::ROIMaskList::getROINames(){
    std::vector<std::string> names{};
    for (const auto& roi : _coll) {
        names.push_back( roi->name );
    }
    return names;
}


std::vector<uint64_t> PreProcess::ROIMaskList::getROICapacities() {
    std::vector<uint64_t> capacities{};
    for (const auto& roi : _coll) {
        capacities.push_back( roi->nones() );
    }
    return capacities;
}