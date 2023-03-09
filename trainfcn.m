load train_data.mat 

%%%%%%%%%%
%Note: In this example, the elution profiles for more-retained pure enantiomer
%are extracted by digitizing Fig. 10 from Mihlbachler et al. [1]

%This code processes the initial and boundary data from experiments to form 
%as appropriate inputs to neural network training code. 

%%%data structures
%csol0 = structure containing initial fluid concentrations for four 
%different experiments 
%csolin = structure containing injection concentrations for four different
%experiments 
%csolout = structure containing elution profiles for four different
%experiments 
%t = temporal data points 
%z = spatial locations across the column

%%%%%%%%%%

%Vector containing different injection concentrations of four more-retained
%pure enantiomer elution experiments
cin=[1.30825;2.6165;3.92475;5.233]; %g/L
tfinal=21; %s

%Note: Experiments 1,2, and 4 in cin vector used for training and 
%experiment 3 as the test case

cinmax=cin./cin(end); %rescaling to (0,1)

t=t./tfinal; %rescaling to (0,1)

[Z, T] = meshgrid(z,t); %spatiotemporal 2D grid


X_sol_1=[Z(:) T(:) cinmax(1).*ones(length(Z(:)),1)];%concatenated input matrix  
%to neural network corresponding to experiment#1

low_bound=min(X_sol_1(:,1:2)); %lower bounds of z and t
up_bound=max(X_sol_1(:,1:2)); %upper bounds of z and t

C0_low_bound=min(cinmax); %lower bound of injection concentration
C0_up_bound=max(cinmax); %upper bound of injection concentration

N_c=5000; %number of collocation points for each experiment #j

X0=[]; %concetenated X matrix representing the initial state
X_lb=[]; %concatenated X matrix representing the inlet boundary
X_rb=[]; %concatenated X matrix representing the outlet boundary
X_c_train=[]; %concatenated X matrix representing collocation points

in_train=[]; %concatenated output matrix containing the initial concentration data
lb_train=[]; %concatenated output matrix containing the injection concentration data
rb_train=[]; %concatenated output matrix containing the elution data


%for loop below stacks initial and boundary labelled data from experiments
%1,2,4 and also concetenates X_c_train matrix of collocation points
for j=[1,2,4]  
    
    xx1=[Z(1,:)' T(1,:)' cinmax(j).*ones(length(Z(1,:)),1)]; %input matrix corresponding to initial condition
    xx2=[Z(:,1) T(:,1) cinmax(j).*ones(length(Z(:,1)),1)]; %input matrix corresponding to inlet boundary
    xx3=[Z(:,end) T(:,end) cinmax(j).*ones(length(Z(:,1)),1)]; %input matrix corresponding to outlet boundary
    
    cc1=csol0.(sprintf('exp%d', j))./cin(j); %initial fluid concentration
    % 
    cc2=csolin.(sprintf('exp%d', j))./cin(j); %injection concentration
    %
    cc3=csolout.(sprintf('exp%d', j))./cin(j); %elution concentration
    %
    x0_train=[xx1;xx2;xx3]; %input matrix corresponding to initial and boundary region
    
    x0_c_train=low_bound + (up_bound-low_bound).*lhsdesign(N_c,2); %generate random collocation points
    x0_c_train=[x0_c_train cinmax(j).*ones(length(x0_c_train),1)]; %create concatenated X matrix for collocation points
    x0_c_train=[x0_c_train;x0_train]; %adding initial and boundary points to collocation matrix

    X0=[X0;xx1]; %stack initial X matrix from experiments 1,2,4
    X_lb=[X_lb;xx2]; %stack inlet X matrix from experiments 1,2,4
    X_rb=[X_rb;xx3]; %stack outlet X matrix from experiments 1,2,4
    X_c_train=[X_c_train;x0_c_train]; %stack collocation X matrix from experiments 1,2,4
    
    in_train=[in_train;cc1]; %stack initial concentration data from experiments 1,2,4
    lb_train=[lb_train;cc2]; %stack injection concentration data from experiments 1,2,4
    rb_train=[rb_train;cc3]; %stack elution concentration data from experiments 1,2,4

   
end

%create index IDs for the loss function in neural network training code
N0_ids=zeros(size(cc1,1),1);
for j=2:3
N0_ids=[N0_ids; (j-1).*ones(size(cc1,1),1)]; 
end

N_b_ids=zeros(size(cc2,1),1);
for j=2:3
N_b_ids=[N_b_ids; (j-1).*ones(size(cc2,1),1)]; 
end

N_c_ids=zeros(size(x0_c_train,1),1);
for j=2:3
N_c_ids=[N_c_ids; (j-1).*ones(size(x0_c_train,1),1)]; 
end

low_bound=[low_bound C0_low_bound];
up_bound=[up_bound C0_up_bound];

save train_chrom.mat X_sol_1 X_c_train in_train lb_train rb_train ...
     X0 X_lb X_rb N0_ids N_b_ids N_c_ids Z T low_bound up_bound

clear