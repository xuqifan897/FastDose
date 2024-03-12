doseFolder = "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/prep_result/doseMatFolder";
offsetsBufferFile = fullfile(doseFolder, "offsetsBuffer.bin");
columnsBufferFile = fullfile(doseFolder, "columnsBuffer.bin");
valuesBufferFile = fullfile(doseFolder, "valuesBuffer.bin");
NonZeroElementsFile = fullfile(doseFolder, "NonZeroElements.bin");
numRowsPerMat = fullfile(doseFolder, "numRowsPerMat.bin");

dimensionFile = "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/prep_output/dimension.txt";
dimension = load(dimensionFile);
return;

fid = fopen(offsetsBufferFile, 'rb');
if fid == -1
    error('Could not open the binary file.');
end
offsetsBuffer = fread(fid, inf, "uint64");
fclose(fid);

fid = fopen(columnsBufferFile, 'rb');
if fid == -1
    error('Could not open the file.');
end
columnsBuffer = fread(fid, inf, "uint64");
fclose(fid);

fid = fopen(valuesBufferFile, 'rb');
if fid == -1
    error('Could not open the file.');
end
valuesBuffer = fread(fid, inf, "float32");
fclose(fid);

fid = fopen(numRowsPerMat, 'rb');
if fid == -1
    error('Could not open the file.');
end
numRowsPerMat = fread(fid, inf, "uint64");

indexii = 1; mats = zeros(numel(columnsBuffer),1);
indexjj = 1;
for ii = 1:numel(numRowsPerMat)
    seg = offsetsBuffer(indexii:indexii+numRowsPerMat(ii));
    mat1 = zeros(seg(end),1);
    mat1(seg(1:end-1)+1) = 1;
    mats(indexjj:indexjj+seg(end)-1) = mat1;
    indexii = indexii+numRowsPerMat(ii)+1;
    indexjj = indexjj+seg(end);
end

rows = cumsum(mats);
M = sparse(columnsBuffer+1,rows,valuesBuffer,prod(phantomdim), sum(numRowsPerMat));



