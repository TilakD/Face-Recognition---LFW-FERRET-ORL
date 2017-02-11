%TILAK AND HARISH 
%DATE: 4:4:2013
clc; clear all; close all;

% =====================PLEASE SPECIFY THE PATH OF THE CUSTOM DATABASE HERE======%
 %=========================Path of the folder uploaded=========================%
custom_database_path = 'C:\Users\Tilak\Documents\MATLAB\Our_Face Recognition\FOR_LFW\lfw_data\resized_database_after_asm\s';
%custom_database_path = 'C:\Users\Tilak\Documents\MATLAB\Our_Face Recognition\FOR_LFW\lfw_data\renamed_custom_lfw\s';
 %========================================================================%
 
 
%%======================Global and Other Parameters======================%%
global classmeanarray totalmeanarray GlobalBestP;
countsum=0;
percentsum=0;
f = fspecial('gaussian',[3 3],0.5);
h= fspecial('log',[3 3],0.5);
b = zeros(35,8);
for i= 1:280
storeface{i} = zeros(1,2500);
end
%%=======================================================================%%
                       %Train and Test Iterations%
%%=======================================================================%%
for x = 1: 10                            %10 Iterations
    disp(strcat('Iteration number:',num2str(x)));
    tic
ttotal = zeros(50,50);
k = 1;
for j = 1:35
%======================Acquire test images Randomly=======================%    
    b(j,:) = randperm(20,8);
    tsum = zeros(50,50);
        for i = 1:8                      %Eight images per subject   

%========================For custom LFW==================================%
face=imread(strcat(custom_database_path,num2str(j),'\',num2str(b(j,i)),'.jpg'));
    
     
   face=rgb2gray(face);
%==========================Preprocessing steps============================%    
  % hgamma = ...
   %vision.GammaCorrector(5,'Correction','gamma');      %Gamma Intensity correction
    %face = step(hgamma, face); 
  %  face = imadjust(face,[],[],(1/2.8)); %Gamma Intensity Correction
  %  face = imfilter(face,f,'replicate'); %Gaussian Blurr
  %  face = imfilter(face,h,'replicate'); %log
  %  face = histeq (face);                %histogram equalization
  % face = logsample(face);
%==========================Feature extraction=============================%    

    dctface = dct2(face);                %DCT of the image
    u=dctface(1:50,1:50);                %Extract 50x50 DCT-coefficients
    
    storeface{k} = reshape(u,1,2500);    %Store in 1x2500 vector
    k = k+1;
    tsum = double(tsum)+double(u);        
        end
ttotal = double(tsum)+double(ttotal);    
avg = (tsum./4);
classmeanarray{j} = avg;                 %Mean of DCT coefficients for 
                                         %each Class(Subject) 
end

avgall = ttotal/280;                        
totalmeanarray = avgall;                 %Mean of DCT coefficints of all 
                                         %Classes
                                         
%%==============================Start BPSO===============================%%                                         

%------------------------Initalization of Parameters----------------------%

NPar = 2500;                             %Number of Dimensional Parameters
NumofParticles = 40;                     %Number of Particles
Velocity = zeros(NumofParticles,NPar);
Position = zeros(NumofParticles,NPar);
Cost = zeros(NumofParticles,1);
LocalBestCost = zeros(NumofParticles,1);
LocalBestPosition = zeros(NumofParticles,NPar);
ff='BPSO_FITNESS_FUNCTION';                           %Fitness function
GlobalBestP = rand(1,NPar);              
GlobalBestC = 0; 

MaxIterations = 30;                      %Number of BPSO iterations
Damp=0.9;                                %Inertial Damping Factor
C1 = 1.619;                              %Cognitive Factor
C2 = .619;                              %Social Factor

%-------------------------Initialization of Particles---------------------%

for i = 1:NumofParticles
    Velocity(i,:) = (rand(1,NPar));
    R = rand(1,NPar);
    Position(i,:) = R < 1./(1 + exp(-Velocity(i,:)));
    Cost(i,:) = feval(ff,Position(i,:),35,NPar);
    LocalBestPosition(i,:) = Position(i,:);
    LocalBestCost(i,:) = Cost(i,:);

    if Cost(i,:) > GlobalBestC
        GlobalBestP = Position(i,:);
        GlobalBestC = Cost(i,:);
    end
end
%----------------------------Start BPSO iterations------------------------%
for t = 1:MaxIterations
    Damp=Damp.^t;
    for i = 1:NumofParticles
        r1 = rand(1,NPar);
        r2 = rand(1,NPar);
        w = rand(1,NPar);
        Velocity(i,:) = Damp*Velocity(i,:) + ...
            r1*C1.*(LocalBestPosition(i,:) - Position(i,:)) + ...
            r2*C2.*(GlobalBestP - Position(i,:));
         
        R = rand(1,NPar);
        Position(i,:) = R < 1./(1 + exp(-Velocity(i,:)));
        Cost(i,:) =feval(ff,Position(i,:),35,NPar);
       
        
        if Cost(i,:) > LocalBestCost(i,:);
            LocalBestPosition(i,:) = Position(i,:);
            LocalBestCost(i,:) = Cost(i,:);
            if Cost(i,:) > GlobalBestC
                GlobalBestP = Position(i,:);
                GlobalBestC = Cost(i,:);
            end
        end   
    end

 end
%--------------------------------End BPSO---------------------------------%

%%==========================Results from BPSO============================%%

count = length(find(GlobalBestP));           %Number of Features selected 
disp('Number of selected features:');
disp(count);
temp(x)=count;

for t= 1:280
    storeface{t}= storeface{t}.*GlobalBestP; %Feature vector for 
end                                          %each Face
trainingtime(x)=toc;
%%=============================Start Testing=============================%%

rec=0;                                       %Recognition Counter
tests=420;                                   %Run test for left out 420
     tic                                        %images
for n=1:tests                       
    c = ceil(n/12); 
    b2 = 1:20;  
    b1 = setdiff(b2,b(c,:));                 %Select images not used in 
                                             %testing stages
                                             
    i = mod(n,12)+(12 * (mod(n,12)==0));        
 
    
%========================For custom LFW==================================%
img=imread(strcat(custom_database_path,num2str(c),'\',num2str(b(i)),'.jpg'));
      

       img=rgb2gray(img);
%-----------------------------Preprocessing-------------------------------%    
   %hgamma = ...
    %vision.GammaCorrector(5,'Correction','gamma');      %Gamma Intensity correction
    %img = step(hgamma, img); 
  % img = imadjust(img,[],[],(1/2.8));
  % img = imfilter(img,f,'replicate');
  % img = imfilter(img,h,'replicate'); %Log
  % img = histeq (img);                %histogram equalization
   %img = logsample(img);
%--------------------------Feature Extraction-----------------------------%
    
    pic=dct2(img);
    pic_dct=reshape(pic(1:50,1:50),1,2500);

%--------------------------Feature Selection------------------------------%    
    
    opt=pic_dct.*GlobalBestP;
    
%-------------Compute Euclidean Distance with each test vector------------%

    d=zeros(1,280);
 
             for p=1:280 
                 r = storeface{p};
     d(p) = sqrt(sum((r-opt).^2));    
             end 
             
     [val,index]=min(d);                   %Minimum of Euclidean Distances
                                           %gives the Matched Vector
%-------------------------------------------------------------------------%
     
     if((ceil(index/8))==c)                %Increment Recognition
     rec=rec+1;                            %Counter if successful  
     end                                   %Recognition
  
end 
%=======================End of One Train-Test iteration===================%
testing=toc;
testOneiteration(x)=(testing/tests);
disp('Recognition rate:');                 %Find Recognition Rate 
percent=(rec/tests)*100;
disp(percent);
percentsum(x)=percent;

end
%%=======================================================================%%
                        %End of Train-test iterations%
%%=======================================================================%%
   
disp('Average number of selected features:')%Find average number of
disp(sum(temp)/max(x));                     %selected features

disp('Average Recognition Rate:')           %Find average of
disp(sum(percentsum)/max(x));               %Recognition rate  
     
disp('Average training time:')
disp(sum(trainingtime)/max(x));

disp('Average testing time:')
disp(sum(testOneiteration)/max(x));
