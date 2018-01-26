%function [ROT_score, VB_score, Diag_score, Size_score, Total_score, Result] = getCompScore_demo(im)
dataset_path = '/home/tianli/face_detection/results_one_person/original/all_background/';
list = dir(dataset_path);

fout = fopen("composition_all_background.txt",'wt');

n_file = size(list,1);

threshold = 0.35;
num = 0;
num_good = 0;
num_bad = 0;

areaBound_small = 0.33;
areaBound_large = 0.69;

sigma_size = 0.33;
sigma_vb = 0.2;
sigma_point = 0.17;

m_EnSize = 0;
m_EnROT = 0;
m_EnVB = 0;
m_EnDiag = 0;
m_sumEn = 0;

m_wtSize = 0.2;
m_wtROT = 1;
m_wtVB = 0.3;
m_wtDiag = 1; % detected diagonal line
m_bDiag = 0;

m_wtROTPt = 0.4;
m_wtROTLn = 0.6;

for ex = 1:n_file
    fname = list(ex).name;
    [~, name, ext] = fileparts(fname);
    if strcmp(name, 'Thumbs')
        continue;
    end
    type = finfo(strcat(dataset_path, fname));

    if strcmp(type, 'im') == 0
        continue;
    end

    im = imread(strcat(dataset_path, fname));
    figure(1); 
    imshow(im);


[height, width, c] = size(im);
if c==1
    im = cat(3, im, im, im);
end

area_image = height * width;
balanceCenter_x = 0.5 * width;
balanceCenter_y = 0.5 * height;




% find prominant lines
endPoints = getLine(im);

[line_value, line_info] = getLineValue(endPoints, width, height);


% find salient objects

map = gbvs(im);
thresholded_map = map.master_map_resized > threshold;
%figure(2); imshow(thresholded_map);
s = regionprops(thresholded_map, 'BoundingBox', 'Area', 'Centroid');
ss = regionprops(thresholded_map, map.master_map_resized, 'Area', 'PixelValues');

multi_ob = 0;
clear temp_area_object;
% salient object??area
if size(s,1) > 1
    for a = 1 : size(s,1)
        temp_area_object(a) = s(a).Area;
    end
    [sorted_area_object, idx] = sort(temp_area_object, 'descend');
    ratio_objects = sorted_area_object(2) / sorted_area_object(1);
   
    if ratio_objects < 0.15
        area_object = sorted_area_object(1);
        bbox_object = s(idx(1)).BoundingBox;
        mean_salval = sum(ss(idx(1)).PixelValues)/ss(idx(1)).Area;
        centroid_object_x = s(idx(1)).Centroid(1);
        centroid_object_y = s(idx(1)).Centroid(2);
    else
        for i = 1 : size(s,1)
            multi_ob = 1;
            area_object(i) = s(i).Area;
            % bbox_object(i) = s(i).BoundingBox;
            mean_salval(i) = sum(ss(i).PixelValues)/ss(i).Area;
            centroid_object_x(i) = s(i).Centroid(1);
            centroid_object_y(i) = s(i).Centroid(2);
        end
    end
else
    area_object = s.Area;
    bbox_object = s.BoundingBox;
    mean_salval = sum(ss.PixelValues)/ss.Area;
    centroid_object_x = s.Centroid(1);
    centroid_object_y = s.Centroid(2);
end


% Salient region size 
ratio_object_image = area_object(1) / area_image;

if (ratio_object_image <= areaBound_small)
    m_EnSize = sqrt((ratio_object_image-0.1)^2);
elseif (ratio_object_image <= areaBound_large) && (ratio_object_image > areaBound_small)
    m_EnSize = sqrt((ratio_object_image-0.56)^2);
elseif (ratio_object_image > areaBound_large)
    m_EnSize = sqrt((ratio_object_image-0.82)^2);
end
m_EnSize = exp(-m_EnSize*m_EnSize/2/sigma_size^2);


% Visual Balance 
x = 0;
y = 0;
d = 0;
weight = 0;
weightSum = 0;

if (multi_ob == 1)
    for i = 1: size(s,1)
        weight = area_object(i) * mean_salval(i);
        x = x + weight * (centroid_object_x(i));
        y = y + weight * (centroid_object_y(i));
        weightSum = weightSum + weight;
    end
else
    weight = area_object * mean_salval;
    x = x + weight * (centroid_object_x);
    y = y + weight * (centroid_object_y);
    weightSum = weightSum + weight;
end

x = x / weightSum - balanceCenter_x;
y = y / weightSum - balanceCenter_y;
x = x / width;
y = y / height;
d = abs(x) + abs(y);
d = exp(-d*d/2/sigma_vb^2);
m_EnVB = d;



% Rule of Thirds 

% Line based ROT 

if (line_info==0) || (line_info == 1) % HORIZONTALLINE or VERTICALLINE
    m_EnROTLn = line_value;
else
    m_EnROTLn = 0;
end

% Point based ROT
dx = 0;
dy = 0;
dist = 0;
weight = 0;
weightSum = 0;
m_EnROTPt = 0;

ptx1 = 1/3 * width;
ptx2 = 2/3 * width;
pty1 = 1/3 * height;
pty2 = 2/3 * height;

if (multi_ob == 1)
    for i = 1 : size(s,1)
        weight = area_object(i) * mean_salval(i);
        dx = min(abs(centroid_object_x(i) - ptx1), abs(centroid_object_x(i) - ptx2));
        dy = min(abs(centroid_object_y(i) - pty1), abs(centroid_object_y(i) - pty2));
        weightSum = weightSum + weight;
        dist = dx / width + dy / height;
        dist = exp(-dist*dist/2/sigma_point^2);
        m_EnROTPt = m_EnROTPt + weight*dist;
    end
else
    weight = area_object * mean_salval;
    dx = min(abs(centroid_object_x - ptx1), abs(centroid_object_x - ptx2));
    dy = min(abs(centroid_object_y - pty1), abs(centroid_object_y - pty2));
    weightSum = weightSum + weight;
    dist = dx / width + dy / height;
    dist = exp(-dist*dist/2/sigma_point^2);
    m_EnROTPt = m_EnROTPt + weight*dist;
end

m_EnROTPt = m_EnROTPt/weightSum;
xx = 0;
if (m_EnROTLn == 0) || (m_EnROTPt == 0)
    m_EnROT = m_EnROTLn + m_EnROTPt;
else
    m_EnROT = 1 / (m_wtROTPt + m_wtROTLn) * (m_wtROTPt * m_EnROTPt + m_wtROTLn * m_EnROTLn);
    xx = m_EnROTPt;
end


% Diagonal Dominance

if (line_info==2) || (line_info == 3) % DIAGONALLINE_HORIZONTAL or DIAGONALLINE_VERTICAL
    m_EnDiag = line_value;
    m_bDiag = 1;
else
    m_EnDiag = 0;
    m_bDiag = 0;
end


% Final composition score
m_sumEn = (m_wtSize*m_EnSize + m_wtROT*m_EnROT + m_wtVB * m_EnVB + m_bDiag * m_wtDiag * m_EnDiag) / (m_wtSize + m_wtROT + m_wtVB + m_bDiag * m_wtDiag);

if (m_sumEn > 0.78)
    Result = 'Good';
elseif (m_sumEn < 0.62)
    Result = 'Bad';
else
    Result = 'Not Bad';
end



    disp(fname);


    disp(sprintf('ROT score : %.2f\n', m_EnROT * m_wtROT));
    disp(sprintf('Size score : %.2f\n', m_EnSize * m_wtSize));
    disp(sprintf('VB score : %.2f\n', m_EnVB * m_wtVB));
    disp(sprintf('Diag score : %.2f\n', m_bDiag * m_EnDiag * m_wtDiag));
    disp(sprintf('total score : %.2f\n', m_sumEn));

    fprintf(fout,"%s ",strcat(dataset_path,fname));
    fprintf(fout, "%.2f ", m_EnROT * m_wtROT);
    fprintf(fout, "%.2f ", m_EnSize * m_wtSize);
    fprintf(fout, "%.2f ", m_EnVB * m_wtVB);
    fprintf(fout, "%.2f ", m_bDiag * m_EnDiag * m_wtDiag);
    fprintf(fout, "%.2f ", m_sumEn);
    fprintf(fout, '\r\n');

end

