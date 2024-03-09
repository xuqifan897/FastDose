#include "rtimages.h"
#include "rtstruct.h"

#include <stdexcept>
#include <dirent.h>
#include <sys/stat.h>
// #include "Utilities/math.h"

// #include "./logconfig.h"
// static LogConfig* logconfig = LogConfig::Instance();

#define MAX_FNAME_CHARS 256

bool PreProcess::closeto(float a, float b, float tolerance) {
    return fabsf(b-a) <= tolerance;
}

PreProcess::RTImage::~RTImage() {
    if (image_count > 0) {
        delete [] slice;
        delete [] data_array;
    }
}

bool PreProcess::RTImage::loadDicomInfo(bool verbose) {
    struct dirent *dp = NULL;
    DIR *dfd = NULL;

    if ((dfd = opendir(dicom_dir.data()) ) == NULL) {
        char buf[MAX_FNAME_CHARS];
        sprintf(buf, "Error: can't open %s\n", dicom_dir.data());
        throw std::runtime_error(buf);
    }

    bool RTImageet_file_found = false;
    image_count = 0;
    while ((dp = readdir(dfd)) != NULL) {
        if ( strcmp(dp->d_name, ".") == 0 || strcmp(dp->d_name, "..") == 0 ) { continue; }
        else if( strstr(dp->d_name,".dcm") != NULL ) {
            if ( importSOPClassUID(dp->d_name) ) { image_count++; }
        }
    }
    closedir(dfd);

    // TODO: this is wasteful to perform type check twice on each file in importSOPClassUID()
    if (verbose) { printf("\n %d RT Images Found. Type: %s\n",image_count,getSOPTypeName().c_str()); }
    if (image_count > 0) {
        RTImageet_file_found = true;
        slice = new SLICE_DATA[image_count];

        dfd = opendir(dicom_dir.data());
        unsigned int current_file = 0;
        while ((dp = readdir(dfd)) != NULL) {
            if ( strcmp(dp->d_name, ".") == 0 || strcmp(dp->d_name, "..") == 0 )
                continue;
            else if (strstr(dp->d_name,".dcm")!=NULL) {
                if ( importSOPClassUID(dp->d_name) ) {
                    slice[current_file].filename = dicom_dir;
                    slice[current_file].filename += "/";
                    slice[current_file].filename += dp->d_name;
                    importInstanceNumber(current_file);
                    importImagePositionPatient(current_file);
                    current_file++;
                }
            }
        }
        closedir(dfd);

        bool keep_sorting = false;
        unsigned int sortCount = image_count;
        do {
            keep_sorting = false;
            for( unsigned int j=1; j<sortCount; j++) {
                if (sop_type == SOPTYPE::CT || sop_type == SOPTYPE::MR || sop_type == SOPTYPE::RTIMAGE) {
                    // sort from foot to head
                    if ( slice[j-1].image_position_patient.z > slice[j].image_position_patient.z ) {
                        keep_sorting = true;
                        SLICE_DATA temp = slice[j-1];
                        slice[j-1] = slice[j];
                        slice[j] = temp;
                    }
                }
            }
            sortCount--;
        } while (keep_sorting && sortCount > 1);

        for( unsigned int j=0; j<image_count; j++) {
            slice[j].instance_number = j;
        }

        importPatientInfo();
    }

    return RTImageet_file_found;
}

bool PreProcess::RTImage::importSOPClassUID(const std::string& fname ) {
    char *filename;
    filename = new char[ dicom_dir.size() + fname.length() + 2];
    sprintf(filename,"%s/%s", dicom_dir.data(), fname.c_str());

    DcmFileFormat format;
    OFCondition status = format.loadFile( filename );
    if (status.bad()) {
        printf("\n Warning: problem reading DICOM file:\n\t%s\n", filename );
        delete [] filename;
        return false;
    }

    bool series_is_rtimage = false;
    OFString ptSOPCLASSUID;
    if (format.getDataset()->findAndGetOFString(DCM_SOPClassUID, ptSOPCLASSUID).good()) {
        if(0 == ptSOPCLASSUID.compare(RTIMAGE_SOP_CLASS_UID)) {
            series_is_rtimage = true;
            sop_type = SOPTYPE::RTIMAGE;
        }
        else if (0 == ptSOPCLASSUID.compare(CTIMAGE_SOP_CLASS_UID)) {
            series_is_rtimage = true;
            sop_type = SOPTYPE::CT;
        }
        else if (0 == ptSOPCLASSUID.compare(MRIMAGE_SOP_CLASS_UID)) {
            series_is_rtimage = true;
            sop_type = SOPTYPE::MR;
        }
    }

    delete []filename;
    return series_is_rtimage;
}

void PreProcess::RTImage::importImagePositionPatient( unsigned int i )
{
    DcmFileFormat format;
    OFCondition status = format.loadFile( slice[i].filename.data() );
    if (status.good())
    {
        DcmDataset* dataset = format.getDataset();
        Float64 imagePOSITION;
        if (dataset->findAndGetFloat64(DCM_ImagePositionPatient, imagePOSITION, 0).good())
            slice[i].image_position_patient.x = imagePOSITION;
        if (dataset->findAndGetFloat64(DCM_ImagePositionPatient, imagePOSITION, 1).good())
            slice[i].image_position_patient.y = imagePOSITION;
        if (dataset->findAndGetFloat64(DCM_ImagePositionPatient, imagePOSITION, 2).good())
            slice[i].image_position_patient.z = imagePOSITION;

        OFString patientPosition;
        if (dataset->findAndGetOFString(DCM_PatientPosition, patientPosition).good())
            slice[i].patient_position = std::string(patientPosition.data());

        std::array<float, 6> imageOrientationPatient;
        for (int ii=0; ii<6; ii++){
            Float64 buf;
            if (dataset->findAndGetFloat64(DCM_ImageOrientationPatient, buf, ii).good())
                imageOrientationPatient[ii] = buf;
        }
        slice[i].image_orientation_patient = imageOrientationPatient;
    }
    else
        printf("\n Error reading DICOM file:\n\t%s\n", slice[i].filename.data() );
}
void PreProcess::RTImage::importInstanceNumber( unsigned int i ) {
    DcmFileFormat format;
    OFCondition status = format.loadFile( slice[i].filename.data() );
    if (status.good())
    {
        if (sop_type == SOPTYPE::CT)
        {
            Sint32 instanceNUMBER;
            if (format.getDataset()->findAndGetSint32(DCM_InstanceNumber, instanceNUMBER).good())
            {
                slice[i].instance_number = instanceNUMBER;
            }
        }
        else if (sop_type == SOPTYPE::MR)
        {
            Float64 sliceLOCATION;
            if (format.getDataset()->findAndGetFloat64(DCM_SliceLocation, sliceLOCATION, 0).good())
            {
                slice[i].slice_location = sliceLOCATION;
            }
        }
    }
    else
        printf("\n Error reading DICOM file:\n\t%s\n", slice[i].filename.data() );
}

void PreProcess::RTImage::importPatientInfo() {
    DcmFileFormat format;
    OFCondition status = format.loadFile( slice[0].filename.data() );
    if (status.bad())
    {
        printf("\n Error reading DICOM file:\n\t%s\n", slice[0].filename.data() );
        return;
    }

    DcmDataset *dataset = format.getDataset();
    OFString ptNAME;
    if (dataset->findAndGetOFString(DCM_PatientName, ptNAME).good())
    {
        pt_name = ptNAME.data();
        //printf("\n PT NAME: %s",pt_name.data());
    }
    OFString ptID;
    if (dataset->findAndGetOFString(DCM_PatientID, ptID).good())
    {
        pt_id = ptID.data();
        //printf("\n PT ID: %s",pt_id.data());
    }
    OFString DICOMDATE;
    if (dataset->findAndGetOFString(DCM_StudyDate,DICOMDATE).good())
    {
        dicom_date = DICOMDATE.data();
        //printf("\n ACQUISITION DATE: %s",dicom_date.data());
    }
    else if (dataset->findAndGetOFString(DCM_SeriesDate,DICOMDATE).good())
    {
        dicom_date = DICOMDATE.data();
       // printf("\n ACQUISITION DATE: %s",dicom_date.data());
    }
    else if (dataset->findAndGetOFString(DCM_AcquisitionDate,DICOMDATE).good())
    {
        dicom_date = DICOMDATE.data();
        //printf("\n ACQUISITION DATE: %s",dicom_date.data());
    }
    OFString ptSERIESDESCRIPTION;
    if (dataset->findAndGetOFString(DCM_SeriesDescription, ptSERIESDESCRIPTION).good())
    {
        pt_series_description = ptSERIESDESCRIPTION.data();
        //printf("\n PT SERIES DESCRIPTION: %s", pt_series_description.data());
    }
    OFString ptSTUDYID;
    if (dataset->findAndGetOFString(DCM_StudyID, ptSTUDYID).good())
    {
        pt_study_id = ptSTUDYID.data();
        //printf("\n PT STUDY ID: %s",pt_study_id.data());
    }
    OFString ptSTUDYINSTANCEUID;
    if (dataset->findAndGetOFString(DCM_StudyInstanceUID, ptSTUDYINSTANCEUID).good())
    {
        pt_study_instance_uid = ptSTUDYINSTANCEUID.data();
        //printf("\n PT STUDY INSTANCE UID: %s",pt_study_instance_uid.data());
    }
    OFString ptSERIESINSTANCEUID;
    if (dataset->findAndGetOFString(DCM_SeriesInstanceUID, ptSERIESINSTANCEUID).good())
    {
        pt_series_instance_uid = ptSERIESINSTANCEUID.data();
        //printf("\n PT SERIES INSTANCE UID: %s",pt_series_instance_uid.data());
    }
    printf("\n");
}


int PreProcess::RTImage::loadRTImageData(bool verbose, bool flipXY) {
    data_min = 9999;
    data_max = -9999;

    OFString pixelRepresentation = 0;
    OFString dataOrdering = 0;
    for (unsigned int i=0; i<image_count; i++) {
        DcmFileFormat format;
        OFCondition status = format.loadFile( slice[i].filename.data() );
        if (!status.good()) {
            char buf[MAX_FNAME_CHARS];
            sprintf(buf, "Error: cannot read Image object (%s)\n", status.text() );
            throw std::runtime_error(buf);
        }

        if (i == 0) {
            data_size.z = image_count;

            Uint16 width,height;
            if (format.getDataset()->findAndGetUint16(DCM_Columns,width).good())
                data_size.x = width;
            if (format.getDataset()->findAndGetUint16(DCM_Rows,height).good())
                data_size.y = height;

            Float64 pixelSPACING;
            if (format.getDataset()->findAndGetFloat64(DCM_PixelSpacing, pixelSPACING, 0).good())
                voxel_size.x = pixelSPACING;
            if (format.getDataset()->findAndGetFloat64(DCM_PixelSpacing, pixelSPACING, 0).good())
                voxel_size.y = pixelSPACING;

            // We need to get z-axis spacing from ImagePositionPatient, not sliceThickness (which is misleading
            // in overlapping/gapped-slice reconstruction)
            if (image_count >= 2) {
              voxel_size.z = (slice[1].image_position_patient.z - slice[0].image_position_patient.z);
            } else {
              if (format.getDataset()->findAndGetFloat64(DCM_SliceThickness, pixelSPACING).good())
                  voxel_size.z = pixelSPACING;
            }

            Float64 sliceThickness;
            if (format.getDataset()->findAndGetFloat64(DCM_SliceThickness, sliceThickness).good())
              slice_thickness = sliceThickness;

            if (!format.getDataset()->findAndGetOFString(DCM_PixelRepresentation, pixelRepresentation).good()) {
                char buf[MAX_FNAME_CHARS];
                sprintf(buf, "Error occured while retreiving DCMElement \"DCM_PixelRepresentation\"\n");
                // throw std::runtime_error(buf);
            }
            OFString temp = 0;
            dataOrdering = OFString("LPS");
            data_order = DATAORDER::LPS;
            if (!format.getDataset()->findAndGetOFString(DCM_Manufacturer, temp).good()) {
                printf("\n Error occured while retreiving DCMElement \"DCM_PixelRepresentation\"\n");
            } else if (temp == "Precision X-Ray") {
                dataOrdering = OFString("RAS");
                data_order = DATAORDER::RAS;
            }


            data_origin = slice[0].image_position_patient;

            Float64 imageORIENT;
            if (format.getDataset()->findAndGetFloat64(DCM_ImageOrientationPatient, imageORIENT, 0).good())
                orient_x.x = imageORIENT;
            if (format.getDataset()->findAndGetFloat64(DCM_ImageOrientationPatient, imageORIENT, 1).good())
                orient_x.y = imageORIENT;
            if (format.getDataset()->findAndGetFloat64(DCM_ImageOrientationPatient, imageORIENT, 2).good())
                orient_x.z = imageORIENT;
            if (format.getDataset()->findAndGetFloat64(DCM_ImageOrientationPatient, imageORIENT, 3).good())
                orient_y.x = imageORIENT;
            if (format.getDataset()->findAndGetFloat64(DCM_ImageOrientationPatient, imageORIENT, 4).good())
                orient_y.y = imageORIENT;
            if (format.getDataset()->findAndGetFloat64(DCM_ImageOrientationPatient, imageORIENT, 5).good())
                orient_y.z = imageORIENT;

            temp.clear();
            if (!format.getDataset()->findAndGetOFString(DCM_PatientPosition, temp).good()) {
                printf("\n Error occured while retreiving DCMElement \"%s\"", DCM_PatientOrientation.toString().c_str());
            }
            orient = std::string(temp.data());


            if (sop_type == SOPTYPE::CT)
            {
                Float64 rescaleINTERCEPT;
                if (format.getDataset()->findAndGetFloat64(DCM_RescaleIntercept, rescaleINTERCEPT).good())
                    rescale_intercept = rescaleINTERCEPT;

                Float64 rescaleSLOPE;
                if (format.getDataset()->findAndGetFloat64(DCM_RescaleSlope, rescaleSLOPE).good())
                    rescale_slope = rescaleSLOPE;

                Float64 windowCENTER;
                if (format.getDataset()->findAndGetFloat64(DCM_WindowCenter, windowCENTER).good())
                    window_center = windowCENTER;

                Float64 windowWIDTH;
                if (format.getDataset()->findAndGetFloat64(DCM_WindowWidth, windowWIDTH).good())
                    window_width = windowWIDTH;
            }

            data_array = new float[ data_size.x * data_size.y * data_size.z ];
            memset( data_array, 0, data_size.x * data_size.y * data_size.z * sizeof(float) );
        }  // END dicom header read

        OFString instanceUID;
        if (format.getDataset()->findAndGetOFString(DCM_SOPInstanceUID, instanceUID).good() ) {
            slice[i].sop_instance_uid = instanceUID.data();
        }
        OFString frameUID;
        if (format.getDataset()->findAndGetOFString(DCM_FrameOfReferenceUID, frameUID).good() ) {
            slice[i].reference_frame_uid = frameUID.data();
        }

        const Uint16* pixelData;
        if (!format.getDataset()->findAndGetUint16Array(DCM_PixelData, pixelData, NULL, 0).good()) {
            char buf[MAX_FNAME_CHARS];
            sprintf(buf, "\n Error: cannot read image pixel data (%s)\n", status.text() );
            throw std::runtime_error(buf);
        }

        for (int yy = 0; yy < data_size.y; yy++) {
            for (int xx = 0; xx < data_size.x; xx++) {
                int p = xx + data_size.x * yy;
                float value = 0.f;
                // handle 2's complement pixel representation
                if (pixelRepresentation == "1") { value = (float)((Sint16*)(pixelData))[p]; }
                else { value = (float)((Uint16*)(pixelData))[p]; }

                // scale data as defined in dicom headers
                if (sop_type == SOPTYPE::CT) {
                    value *= rescale_slope;
                    value += rescale_intercept;
                }
                // flip to dicom standard LPS data ordering (x+ toward pt. left; y+ toward posterior; z+ toward cranial) instread of RAS
                int x, y;
                if (flipXY) {
                    x = data_size.x - xx - 1;
                    y = data_size.y - yy - 1;
                } else {
                    x = xx;
                    y = yy;
                }

                setArrayVoxel( x, y, i, value );
                // bounds check
                if (value > data_max) data_max = value;
                if (value < data_min) data_min = value;
            }
        }
    }

    // interpret data orientation and give accurate description of patient orientation
    char data_orientation[4] = "UNK";
    if (closeto(orient_x.x, 1.0) && closeto(orient_y.y, 1.0)) {
      strcpy(data_orientation, "HFS");
    } else if (closeto(orient_x.x, -1.0) && closeto(orient_y.y, -1.0)) {
      strcpy(data_orientation, "HFP");
    }

    if (verbose) {
        printf("DICOM ATTRIBUTES\n----------------\n");
        printf("Image Size:            %d x %d x %d\n",data_size.x,data_size.y,data_size.z);
        printf("Voxel Spacing [mm]:    %1.3f x %2.3f x %2.3f\n",voxel_size.x,voxel_size.y,voxel_size.z);
        printf("Image Position [mm]:   %1.3f x %2.3f x %2.3f\n",data_origin.x,data_origin.y,data_origin.z);
        printf("Pixel Representation:  %s\n", pixelRepresentation == "1" ? "Signed Integer" : "Unsigned Integer");
        printf("Data Ordering:         %s%s\n", dataOrdering.c_str(), flipXY ? " (180 deg Z-axis rotation)" : "");
        // printf("Patient Position:      %s\n", orient.c_str());
        printf("Image Orientation X:   [%0.3f, %0.3f, %0.3f]\n",orient_x.x,orient_x.y,orient_x.z);
        printf("Image Orientation Y:   [%0.3f, %0.3f, %0.3f]\n",orient_y.x,orient_y.y,orient_y.z);
        printf("Patient Orientation:   %s\n", data_orientation);
        // printf("Window Center: %3.3f\n", window_center);
        // printf("Window Width: %3.3f", window_width);
        printf("Rescale Correction:    (%0.2f)*X + %0.2f\n", rescale_slope, rescale_intercept);
        printf("Data Range:            %0.3f --> %0.3f\n", data_min, data_max);
        printf("\n");
    }
    return 1;
}

int PreProcess::RTImage::saveRTImageData(const std::string& outpath, float *newData, bool anonymize_switch ) {
    for (int i=0; i<data_size.z; i++)
    {
        DcmFileFormat format;
        OFCondition status = format.loadFile( slice[i].filename.data() );
        if (status.good())
        {
            DcmDataset *dataset = format.getDataset();

            if (anonymize_switch)
                anonymize( dataset );

            time_t rawtime;
            struct tm * timeinfo;
            char buffer[64];
            time (&rawtime);
            timeinfo = localtime (&rawtime);
            strftime (buffer,32,"%G%m%d",timeinfo);
            dataset->putAndInsertString(DCM_AcquisitionDate,buffer).good();
            dataset->putAndInsertString(DCM_SeriesDate,buffer).good();

            Uint16 width,height,images;
            width = data_size.x;
            height = data_size.y;
            images = data_size.z;
            dataset->putAndInsertUint16(DCM_Columns,width).good();
            dataset->putAndInsertUint16(DCM_Rows,height).good();
            dataset->putAndInsertUint16(DCM_ImagesInAcquisition,images).good();
            //printf("\n Output Data Dimensions: %d x %d x %d", width, height, images );

            Float64 vz;
            vz = voxel_size.z;
            dataset->putAndInsertFloat64(DCM_SliceThickness,vz).good();

            OFString vxy;
            char pixelSpacing[32];
            sprintf(pixelSpacing,"%3.3f\\%3.3f",voxel_size.x,voxel_size.y);
            vxy.assign( (const char*)pixelSpacing );
            dataset->putAndInsertOFStringArray(DCM_PixelSpacing,vxy,OFTrue).good();

            OFString vxyz;
            char dataOrigin[32];
            sprintf(dataOrigin,"%3.3f\\%3.3f\\%3.3f",data_origin.x,data_origin.y,data_origin.z);
            vxyz.assign( (const char*)dataOrigin );
            dataset->putAndInsertOFStringArray(DCM_ImagePositionPatient,vxyz,OFTrue).good();

            Sint32 instanceNUMBER;
            instanceNUMBER = slice[i].instance_number;
            dataset->putAndInsertSint32(DCM_InstanceNumber, instanceNUMBER).good();

            dataset->putAndInsertString(DCM_SOPInstanceUID, slice[i].sop_instance_uid.data());

            Float64 rescaleINTERCEPT;
            rescaleINTERCEPT = rescale_intercept;
            dataset->putAndInsertFloat64(DCM_RescaleIntercept, rescaleINTERCEPT);

            Float64 rescaleSLOPE;
            rescaleSLOPE = rescale_slope;
            dataset->putAndInsertFloat64(DCM_RescaleSlope, rescaleSLOPE);

            Uint16 *pixelDataOut;
            pixelDataOut = new Uint16[width*height];
            for (int y=0; y<data_size.y; y++)
                for (int x=0; x<data_size.x; x++)
                {
                    int p = x + data_size.x*y;
                    Uint16 temp = 0;
                    //temp = (Uint16)((getArrayVoxel(x,y,i) - rescale_intercept) / rescale_slope );
                    if (sop_type == SOPTYPE::CT)
                        temp = (Uint16)((newData[p + i*data_size.x*data_size.y] - rescale_intercept) / rescale_slope );
                    else if (sop_type == SOPTYPE::MR)
                        temp = (Uint16)newData[p + i*data_size.x*data_size.y];
                    pixelDataOut[p] = temp;
                }

            status = dataset->putAndInsertUint16Array(DCM_PixelData, pixelDataOut, width*height, OFTrue);
            delete []pixelDataOut;

            mkdir(outpath.c_str(),S_IRWXU|S_IRWXG|S_IRWXO);

            if (status.good())
            {
                char outfilename[512];
                sprintf(outfilename,"%s/NewDICOM_Image_Data_%d.dcm",outpath.c_str(),i+1);
                //printf("  Saving %s ... \r",outfilename);
                status = format.saveFile( outfilename );
                if (status.bad())
                    printf("Error: cannot write DICOM file ( %s )",status.text() );
            }
        }
        else
            printf("\n Error: cannot read Image object (%s)\n", status.text() );
    }
    return 1;
}

void PreProcess::RTImage::saveRTImageData(const std::string& outpath, bool anonymize_switch ) {
    for (unsigned int i=0; i<image_count; i++)
    {
        DcmFileFormat format;
        OFCondition status = format.loadFile( slice[i].filename.c_str() );
        if (status.good())
        {
            DcmDataset *dataset = format.getDataset();

            if (anonymize_switch)
                anonymize( dataset );

            time_t rawtime;
            struct tm *timeinfo;
            char buffer[32];
            time (&rawtime);
            timeinfo = localtime (&rawtime);
            strftime(buffer,32,"%G%m%d",timeinfo);
            //format.getDataset()->putAndInsertString(DCM_AcquisitionDate,buffer).good();
            format.getDataset()->putAndInsertString(DCM_SeriesDate,buffer).good();
            fflush(stdout);

            mkdir(outpath.c_str(),S_IRWXU|S_IRWXG|S_IRWXO);

            char *outfilename = new char[ outpath.length() + 32 ];
            sprintf(outfilename,"%s/New_DICOM_Image_Data_%d.dcm",outpath.c_str(),i+1);
            status = format.saveFile( outfilename );
            if (status.bad())
                printf("Error: cannot write DICOM file ( %s )", status.text() );
        }
        else
            printf("\n Error: cannot read Image object (%s)\n", status.text() );
    }
    return;
}

void PreProcess::RTImage::anonymize(DcmDataset *dataset) {
    dataset->findAndDeleteElement(DCM_AccessionNumber, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_ReferringPhysicianName, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_ReferringPhysicianAddress, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_ReferringPhysicianTelephoneNumbers, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_ReferringPhysicianIdentificationSequence, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_PhysiciansOfRecord, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_InstitutionName, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_InstitutionAddress, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_InstitutionCodeSequence, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_PatientName, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_PatientID, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_IssuerOfPatientID, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_TypeOfPatientID, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_IssuerOfPatientIDQualifiersSequence, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_PatientBirthDate, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_PatientBirthTime, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_PatientSex, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_PatientInsurancePlanCodeSequence, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_PatientPrimaryLanguageCodeSequence, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_PatientPrimaryLanguageModifierCodeSequence, OFTrue, OFTrue);
//    dataset->findAndDeleteElement(DCM_OtherPatientIDs, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_OtherPatientNames, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_OtherPatientIDsSequence, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_PatientBirthName, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_PatientAge, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_PatientSize, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_PatientSizeCodeSequence, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_PatientWeight, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_PatientAddress, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_PatientMotherBirthName, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_MilitaryRank, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_BranchOfService, OFTrue, OFTrue);
  //  dataset->findAndDeleteElement(DCM_MedicalRecordLocator, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_MedicalAlerts, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_Allergies, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_CountryOfResidence, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_RegionOfResidence, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_PatientTelephoneNumbers, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_EthnicGroup, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_Occupation, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_SmokingStatus, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_AdditionalPatientHistory, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_PregnancyStatus, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_LastMenstrualDate, OFTrue, OFTrue);
    dataset->findAndDeleteElement(DCM_PatientReligiousPreference, OFTrue, OFTrue);
}


PreProcess::ArrayProps PreProcess::getROIExtents(const StructureSet& roi,
    const FrameOfReference& frame, bool verbose) {
    float3 min_coord{ std::numeric_limits<float>::max(),  std::numeric_limits<float>::max(),  std::numeric_limits<float>::max()};
    float3 max_coord{std::numeric_limits<float>::min(), std::numeric_limits<float>::min(), std::numeric_limits<float>::min()};

    // slice iterator
    uint mark = 0;
    for (int c=0; c<(int)roi.sub_cntr_count; ++c) {
        // point iterator
        for (int i=0; i<(int)roi.sub_cntr_points_count[c]; ++i) {
            // get coord
            float x = roi.points[3*(i+mark)  ];
            float y = roi.points[3*(i+mark)+1];
            float z = roi.points[3*(i+mark)+2];
            // if (verbose) { printf("  c:%d, i:%d, x:%f, y:%f, z:%f\n", c, i, x, y, z); }

            // update limits
            if (x < min_coord.x) { min_coord.x = x; }
            if (y < min_coord.y) { min_coord.y = y; }
            if (z < min_coord.z) { min_coord.z = z; }
            if (x > max_coord.x) { max_coord.x = x; }
            if (y > max_coord.y) { max_coord.y = y; }
            if (z > max_coord.z) { max_coord.z = z; }
        }
        mark += roi.sub_cntr_points_count[c];
    }

    // convert to indices
    int3 start, end;
    start.x = floor( (0.1f*min_coord.x - frame.start.x) / (frame.spacing.x) );
    start.y = floor( (0.1f*min_coord.y - frame.start.y) / (frame.spacing.y) );
    start.z = floor( (0.1f*min_coord.z - frame.start.z) / (frame.spacing.z) );
    end.x   = ceil(  (0.1f*max_coord.x - frame.start.x) / (frame.spacing.x) );
    end.y   = ceil(  (0.1f*max_coord.y - frame.start.y) / (frame.spacing.y) );
    end.z   = ceil(  (0.1f*max_coord.z - frame.start.z) / (frame.spacing.z) );

    // bounds checking
    if (start.x < 0) { start.x = 0; }
    if (start.y < 0) { start.y = 0; }
    if (start.z < 0) { start.z = 0; }
    if (start.x > (int)frame.size.x) { start.x = frame.size.x; }
    if (start.y > (int)frame.size.y) { start.y = frame.size.y; }
    if (start.z > (int)frame.size.z) { start.z = frame.size.z; }
    if (end.x < 0) { end.x = 0; }
    if (end.y < 0) { end.y = 0; }
    if (end.z < 0) { end.z = 0; }
    if (end.x > (int)frame.size.x) { end.x = frame.size.x; }
    if (end.y > (int)frame.size.y) { end.y = frame.size.y; }
    if (end.z > (int)frame.size.z) { end.z = frame.size.z; }

    if (verbose) {
        printf("Frame.size: (%d, %d, %d)  ||  Frame.start: (%6.2f, %6.2f, %6.2f)  ||  Frame.spacing: (%6.2f, %6.2f, %6.2f)\n",
                frame.size.x, frame.size.y, frame.size.z,
                frame.start.x, frame.start.y, frame.start.z,
                frame.spacing.x, frame.spacing.y, frame.spacing.z
                );
        printf("Min Coords: (%6.2f, %6.2f, %6.2f)  ||  Max Coords: (%6.2f, %6.2f, %6.2f)\n",
                min_coord.x, min_coord.y, min_coord.z, max_coord.x, max_coord.y, max_coord.z);
        printf("Min Indices: (%d, %d, %d)  ||  Max Indices: (%d, %d, %d)\n",
                start.x, start.y, start.z, end.x, end.y, end.z);
    }

    // calculate FrameOfReference
    ArrayProps props {};
    props.size = frame.size;
    props.crop_start = make_uint3(start);
    props.crop_size = make_uint3(end-start);
    if (verbose) {
        printf("crop_start: (%d, %d, %d)  ||  crop_size: (%d, %d, %d)\n",
                props.crop_start.x, props.crop_start.y, props.crop_start.z,
                props.crop_size.x, props.crop_size.y, props.crop_size.z);
    }
    return props;
}