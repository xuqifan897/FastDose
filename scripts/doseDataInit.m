%% This file initializes the structure `dose_data`

doseMatFolder = "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/prep_result/doseMatFolder";
dose_data_file = fullfile(doseMatFolder, "dose_data.h5");

% file = H5F.open(dose_data_file, "H5F_ACC_RDONLY", "H5P_DEFAULT");

file_info = h5info(dose_data_file);
for i = 1:length(file_info.Datasets)
    disp(file_info.Datasets(i).Name);
end

N_beamlets = h5read(dose_data_file, "/N_beamlets");
column_labels = h5read(dose_data_file, "/column_labels");
gantry_rot_rad = h5read(dose_data_file, "/gantry_rot_rad");
couch_rot_rad = h5read(dose_data_file, "/couch_rot_rad");
fmap_dims = h5read(dose_data_file, "/fmap_dims");

num_beams = size(N_beamlets, 1);
for kk = 1 : num_beams
    dose_data.beam_metadata(kk).N_beamlets = N_beamlets(kk);
    dose_data.beam_metadata(kk).beam_specs = struct();
    dose_data.beam_metadata(kk).beam_specs.gantry_rot_rad = gantry_rot_rad(kk);
    dose_data.beam_metadata(kk).beam_specs.couch_rot_rad = couch_rot_rad(kk);
    dose_data.beam_metadata(kk).beam_specs.fmap_dims = fmap_dims(:, kk);

    beam_specs = struct('gantry_rot_rad', gantry_rot_rad(kk), 'couch_rot_rad', couch_rot_rad(kk), 'fmap_dims', fmap_dims(:, kk));
    dose_data.beam_metadata(kk) = struct('N_beamlets', N_beamlets(kk), ...
        'beam_specs', beam_specs);
end
dose_data.column_labels = transpose(column_labels);
