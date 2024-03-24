function h=whisker_boxplot(x,y,color,varargin)
% function h=whisker_boxplot(x,y,color,varargin)
% x= HorzX value for plot (i.e. group #)
% y= data (single vector or DATA x GROUP vector)
%    data can be pre-sorted for meaning, e.g. youngest to oldest.
% color= {[0 0 0]} default
% vargin pairs:
% 'notch'  = { [0], 1 }; display notches: 0= none; 1= symmetrical CI.
% 'width'  = [.3]; half width of box plot: range .1:.4 (.5 will touch )
% 'bub'    = {true, [false]}; display ALL data points, making a 'bubble' plot
%            bubble plots are shifted by data order(intra-sample scaled), or 'shiftxs' (inter-data sample scaled)
%            pre-order data by how you want is shifted/spread (i.e. by age)
% 'marker' = set an alternative marker for outliers or bubble plots.
% 'shiftxs'= how much to shift the each X value for each data point
%            (e.g. [-width:width] ), one for each data poiny in y.
% 'sat'    = {1:0} [.75]; saturation of box facecolor: 1= full color, .5 is half tone, 0 = white
% 'prop'   = proportional width based on sample size. The value of 'prop' should
%            equal the largest (or expected) sample size. thus actual_width =  * sum(~isnan(y))/prop
% 'horz'   = {true,[false]}; plot horizontally
%
% h = handles for the objects in each plot
%    h(x,1) = fill 'facecolor' (sat main color) & 'edgecolor' (main color)
%    h(x,2) = median line (darkened main color)
%    h(x,3) = central line (main color)
%    h(x,4) = extreme lines (main color)
%    h(x,5) = outlier points (main color)
%    h(x,6) = if requested, individual points (main color)
%
%
% see: adjusted_boxplot
% see: violin_plot
%
% 2019-may-21 Brian Coe coe@queensu.ca

if nargin==0
    help(mfilename);% standard bcoe format
    return
end

if nargin==1
    y=x;
    x=1:min(size(y));
end
if isempty(y)
    h=[];
    return
end

if all(isnan(y))
    h=[];
    return
end
[o,g] = size(y);% observations (row) x groups  (col)
if g>o % can't recommend a case where one has more groups than observations.
    y=y';
    [o,g] = size(y); %#ok<ASGLU>
end

if isempty(x)
    x=1:g;
end

if ~exist('color','var')
    color={[0 0 0]};
end
if isempty(color)
    color={[0 0 0]};
end
if ischar(color)
    error('please use RGB triplet colors: [0 0 0]')
end
if ~iscell(color)
    color={color};
end

% if length(x)>length(color)
%     color=repmat(color{1},1,length(x));
% end
if ~isempty(varargin)
    if ~mod(length(varargin),2)==0
        error('if ~mod(length(varargin),2)==0')
    end
    for ii=1:2:length(varargin)
        switch class(varargin{ii+1})
            case 'char'
                eval(sprintf('%s=''%s'';',varargin{ii},varargin{ii+1}))
            case {'logical','double','single','int8','uint8','int16','uint16'}
                eval(sprintf('%s=varargin{ii+1};',varargin{ii}))
            otherwise
                warning('varargin class not stipulated')
                drawnow
        end
    end
end

if ~exist('notch','var')
    notch=0; 
else % box plots has notches
    notch=notch==true; %#ok<NODEF>
end
if ~exist('width','var')
    width=.3; % radius of box plots
end
% option of width being proportional to an expected sample size...
if ~exist('prop','var')
    prop=length(y);
else
    prop=double(prop); %#ok<NODEF>
end

if ~exist('shiftxs','var')
    shiftxs=[];
end
if ~exist('bub','var')
    bub=false;
else
    bub=bub==true; %#ok<NODEF>
end
if ~exist('marker','var')
    marker='o';
end
if ~exist('SDC','var')
    SDC=20;
else
    SDC=double(SDC);%#ok<NODEF>
end

if ~exist('horz','var')
    horz=false;
else
    horz=horz==true; %#ok<NODEF>
end

if ~exist('sat','var')
    sat=.75; % box plots are 75% saturated
else
    sat=double(sat); %#ok<NODEF>
end

hold on

W1p5=1.5;% this is standard box plot stuff;
lWidth=1.25;% line width;

h=zeros(g,5);
MC=zeros(1,g);
for ii =1:g
    colorM=color{ii}/2;
    colorO=color{ii};
    if sat>1
        warning('saturation over range');
        colorI=color{ii};
    elseif sat<0
        warning('saturation under range');
        colorI=[1 1 1];
    else
        colorI=1-(1-color{ii})*sat;
    end    
    
    X=x(ii);% indices may not match group number (plot position)
    data=single(y(:,ii));
    data(isnan(data))=[];
    nx = length(data);
    % option of width being proportional to sample size...
    widthp = length(data)/prop*width; % radius of box plots adjusted by sample size
    
    % data should be pre-sorted for meaning, e.g. youngest to oldest.
    if isempty(shiftxs)
        shiftx=X+linspace(-widthp,widthp,length(data));
    else
        shiftx=X+shiftxs(~isnan(y(:,ii)),g);
    end
    Q = double(prctile(data,[25 50 75]));
    IQR=(Q(3)-Q(1));
    int_range=[Q(1)-W1p5*IQR;  Q(3)+W1p5*IQR];
    
    hiOut=data>int_range(2);
    loOut=data<int_range(1);
    maxIn=max(data(~hiOut));
    minIn=min(data(~loOut));
    
    CI=(1.57*IQR/sqrt(nx))*[-1; 1]+ Q(2); % default according to matlab
    if notch>0
        xx=X+[-widthp -widthp -widthp/2 -widthp -widthp  widthp widthp widthp/2 widthp widthp ];
        mm=X+[-widthp widthp]/2;
        yy= [Q(1)  CI(1) Q(2)  CI(2)  Q(3) Q(3)  CI(2) Q(2)  CI(1) Q(1)];
    else
        xx=X+[-widthp  -widthp -widthp  widthp  widthp widthp ];
        mm=X+[-widthp widthp];
        yy=Q([ 1 2 3 3 2 1]);
    end
    
    if horz
        h(ii,3)=plot([minIn maxIn],[X X], '-','color',colorO,'LineWidth',lWidth);
        h(ii,1)=fill(yy,xx,colorO,'FaceColor',colorI,'EdgeColor',colorO,'LineWidth',lWidth);
        h(ii,4)=plot([maxIn maxIn nan minIn minIn],[mm nan mm],'color',colorO,'LineWidth',lWidth);
        if any(hiOut|loOut)
            h(ii,5)=plot(data(hiOut|loOut),shiftx(hiOut|loOut),'+','color',colorO*.9);
        end% if any(hiOut|loOut)
        if bub
            h(ii,6)=plot(data,shiftx,'o','color',colorO*.9,'marker',marker);
            if any(hiOut|loOut)
                set(h(ii,5),'marker','.');
            end
        end % if bub
        h(ii,2)=line(Q([2 2]),mm,'color',colorM,'LineWidth',lWidth*2); % median line should plot last
    else
        h(ii,3)=plot([X X], [minIn maxIn],'-','color',colorO,'LineWidth',lWidth);
        h(ii,1)=fill(xx,yy,colorO,'FaceColor',colorI,'EdgeColor',colorO,'LineWidth',lWidth);
        h(ii,4)=plot([mm nan mm],[maxIn maxIn nan minIn minIn],'color',colorO,'LineWidth',lWidth);
        if any(hiOut|loOut)
            h(ii,5)=plot(shiftx(hiOut|loOut),data(hiOut|loOut),'+','color',colorO*.9);
        end% if any(hiOut|loOut)
        if bub
            h(ii,6)=plot(shiftx,data,'o','color',colorO*.9,'marker',marker);
            if any(hiOut|loOut)
                set(h(ii,5),'marker','.');
            end
        end % if bub
        h(ii,2)=line(mm,Q([2 2]),'color',colorM,'LineWidth',lWidth*2); % median line should plot last
    end
    
end
end % function

%%




