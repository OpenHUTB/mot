
function y = vsdx_pdf_crop(varargin)

viso_files = dir("*.vsdx");

if(isempty(varargin))
    % translate the lastest viso file
    viso_datenum = [viso_files.datenum];
    [~, latest_i] = max(viso_datenum);
    translate_ids = latest_i;
else
    % translate all viso file
    translate_ids = range(length(viso_files));
end

for i = 1 : length(translate_ids)
    viso_file = viso_files(translate_ids(i)).name;
    pdf_name = strcat( viso_file(1:end-5), ".pdf");

    viso2pdf_cmd = strcat("viso2pdf.exe ", viso_file);
    % translate viso to pdf in windows( writen with C#)
    [translate_status, ~] = system(viso2pdf_cmd);
    disp(fprintf("Translate (%s to %s) status: %d", viso_file , pdf_name, translate_status));

    pdf_crop_cmd = strcat("pdfcrop ", pdf_name, " ", pdf_name);
    % crop PDF file to remove the white border (call latex command)
    [crop_status, ~] = system(pdf_crop_cmd);
    disp(fprintf("Crop PDF (%s) status: %d", pdf_name, crop_status));
end


end