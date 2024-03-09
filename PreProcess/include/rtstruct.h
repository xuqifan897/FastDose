#ifndef __RTSTRUCT_H__
#define __RTSTRUCT_H__
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "dcmtk/dcmdata/dctk.h"

#define RTSTRUCT_SOP_CLASS_UID "1.2.840.10008.5.1.4.1.1.481.3"

namespace PreProcess {
    class RTStruct
    {
    public:
        RTStruct  ();
        ~RTStruct ();

        class CNTR_DATA
        {
        public:
            float *points;
        };

        class ROI_DATA
        {
        public:
            int roi_number;
            std::string roi_name;
            int3 roi_rgb_color;
            unsigned int sub_cntr_count;
            bool load_data;
            unsigned int *sub_cntr_points_count;
            unsigned int total_points_count;
            float3 range_min;
            float3 range_max;
            CNTR_DATA *sub_cntr_data;
        };

        bool loadDicomInfo(bool verbose=false);
        void loadRTStructInfo(bool verbose=false);
        int  loadRTStructData( int r, bool verbose=false);
        void chooseContours();
        bool importSOPClassUID( char *buffer );
        void importPatientInfo();

        void copyROI( int r, ROI_DATA *roi_copy, int c );
        void freeROI( ROI_DATA *roi_copy, int c );
        void anonymize( DcmDataset *dataset );
        void saveRTStructData ( const char *outpath, ROI_DATA *new_roi_data, unsigned int new_roi_count, bool anonymize_switch );
        void saveRTStructData ( const char *outpath, bool anonymize_switch );

        unsigned int getNumberOfROIs()
        {
            return roi_count;
        };
        char*  getDicomDirectory()
        {
            return (char*)dicom_dir.data();
        };
        char*  getDicomFilename()
        {
            return (char*)dicom_full_filename.data();
        };

        float3 getSubCntrPoint( unsigned int r, unsigned int s, unsigned int p );
        int    getROINumber( unsigned int r )
        {
            return roi_array[r].roi_number;
        };
        char*  getROIName( unsigned int r )
        {
            return (char*)roi_array[r].roi_name.data();
        };
        int3    getROIColor( unsigned int r )
        {
            return roi_array[r].roi_rgb_color;
        };
        unsigned int    getROISubCntrCount( unsigned int r )
        {
            return roi_array[r].sub_cntr_count;
        };
        unsigned int    getROISubCntrPointCount( unsigned int r, unsigned int s )
        {
            return roi_array[r].sub_cntr_points_count[s];
        };
        bool   getROILoadDataSwitch( unsigned int c )
        {
            return roi_array[c].load_data;
        };
        float* getROISubCntrPoints( unsigned int r, unsigned int s )
        {
            return roi_array[r].sub_cntr_data[s].points;
        };
        unsigned int    getROITotalPointsCount( unsigned int r )
        {
            return roi_array[r].total_points_count;
        };

        void   setDicomFilename( char *buffer );
        void   setDicomDirectory( const char *buffer );
        void   setSubCntrPoint( unsigned int r, unsigned int s, unsigned int p, float v_x, float v_y, float v_z );

    protected:
        std::string dicom_dir;
        std::string dicom_date;
        std::string dicom_full_filename;
        std::string pt_series_description;
        std::string pt_name;
        std::string pt_id;
        std::string pt_study_id;
        std::string pt_sop_class_uid;
        std::string pt_sop_instance_uid;
        std::string pt_study_instance_uid;
        std::string pt_series_instance_uid;

        bool rtstruct_file_found;

        unsigned int roi_count;
        ROI_DATA *roi_array;
    };


    struct StructureSet {
        unsigned int       sub_cntr_count = 0; // nslices in "CLOSED_PLANAR" DICOM storage mode
        std::vector<uint>  sub_cntr_points_count;  // list of coord. counts for each sub_cntr
        unsigned int       total_points_count = 0;
        std::vector<float> points;
    };


    int getROIIndex(RTStruct& rtstruct, const std::string& search,
        bool exact=false, bool verbose=false);

    bool loadStructureSet(StructureSet& roi, RTStruct& rtstruct,
        uint roi_idx, bool verbose=false);
}

#endif