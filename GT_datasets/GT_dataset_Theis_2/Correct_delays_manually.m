

delays = [0.0418 0.0418 -1.548 -0.8785 -0.0418 -0.0418 -0.0418 0.3765 0.3765 0.3765 0.3765];

numel(delays)

fileList = dir('*mini.mat');

for j = 1:numel(fileList)
    
    load(fileList(j).name);
    
    
    numel(CAttached)
    
    CAttached{1}.events_AP = CAttached{1}.events_AP + delays(j)*1e4;
    
    save([fileList(j).name(1:end-8),'corrected_mini.mat'],'CAttached')
    
end