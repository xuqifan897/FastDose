doseFolder = "/data/qifan/projects/FastDoseWorkplace/BOOval/LUNG/prep_result/doseMatFolder";
offsetsBufferFile = fullfile(doseFolder, "offsetsBuffer.bin");
columnsBufferFile = fullfile(doseFolder, "columnsBuffer.bin");
valuesBufferFile = fullfile(doseFolder, "valuesBuffer.bin");
NonZeroElementsFile = fullfile(doseFolder, "NonZeroElements.bin");
numRowsPerMat = fullfile(doseFolder, "numRowsPerMat.bin");
phantomDim = [220, 220, 149];

numVoxels = prod(phantomDim);

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

fid = fopen(NonZeroElementsFile, 'rb');
if fid == -1
    error('Could not open the file.');
end
NonZeroElements = fread(fid, inf, "uint64");

fid = fopen(numRowsPerMat, 'rb');
if fid == -1
    error('Could not open the file.');
end
numRowsPerMat = fread(fid, inf, "uint64");


numMatrices = size(numRowsPerMat, 1);
cumuNnz = zeros(numMatrices, 1);
cumuNumRows = zeros(numMatrices, 1);
for i = 2:numMatrices
    cumuNnz(i) = cumuNnz(i-1) + NonZeroElements(i-1);
    cumuNumRows(i) = cumuNumRows(i-1) + numRowsPerMat(i-1) + 1;
end

matrices = cell(numMatrices, 1);
for i = 1:numMatrices
    localNnz = NonZeroElements(i);
    localNumRows = numRowsPerMat(i);
    local_cumuNnz = cumuNnz(i);
    local_cumuNumRows = cumuNumRows(i);

    m_offsets = offsetsBuffer(local_cumuNumRows + 1: local_cumuNumRows + 1 + localNumRows);
    m_columns = columnsBuffer(local_cumuNnz + 1: local_cumuNnz + 1 + localNnz - 1);
    m_values = valuesBuffer(local_cumuNnz + 1: local_cumuNnz + 1 + localNnz - 1);
    
    m_rows = m_columns;
    row_offset = 1;
    for j = 1:localNumRows
        idx_begin = m_offsets(j) + 1;
        idx_end = m_offsets(j + 1);
        m_rows(idx_begin: idx_end) = j;
    end
    m_columns = m_columns + 1;
    matrices{i} = sparse(m_columns, m_rows, m_values, numVoxels, localNumRows);

    whos matrices{1}
    break;
end