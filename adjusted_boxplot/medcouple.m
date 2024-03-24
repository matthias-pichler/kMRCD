function MC=medcouple(datain)
    % FUNCTION MC=medcouple(datain)
    % coded brian coe 2019jun18 coe@queensu.ca
    % datain        : the data coming in should be organized by: observations (row) x groups/individuals (col)
    % MC(1,group)   : will be returned
    %
    % calculations are based on formulas 2.1, 2.2, & 2.3 from [1].
    %
    % wikipedia: In statistics, the medcouple is a robust statistic that measures the
    % skewness of a univariate distribution[1]. It is defined as a scaled
    % median difference of the left and right half of a distribution. Its
    % robustness makes it suitable for identifying outliers in adjusted
    % boxplots[2][3].
    %
    %1 = G. Brys; M. Hubert; A. Struyf (November 2004). "A Robust Measure of
    %       Skewness". Journal of Computational and Graphical Statistics. 13
    %       (4): 996�1017.
    %2 = M. Hubert; E. Vandervieren (2008). "An adjusted boxplot for skewed
    %       distributions". Computational Statistics and Data Analysis. 52 (12):
    %       5186�5201. doi:10.1016/j.csda.2007.11.008.
    %3 = Pearson, Ron (February 6, 2011). "Boxplots and Beyond � Part II:
    %       Asymmetry". exploringdata.blogspot.ca. Retrieved April 6, 2015.
    %
    
    plotit=false;
    %plotit=true; % TRUE for plots (best with breakpoints at each 'drawnow')
    
    
    if nargin==0
        help(mfilename);% standard bcoe format
        return
    end
    
    [o,g] = size(datain);% observations (row) x groups (col)
    if g>o % can't recommend a case where one has more groups than observations.
        datain=datain';
        [o,g] = size(datain); %#ok<ASGLU>
    end
    MC=zeros(1,g);
    m =nanmedian(datain); % get medians for each group
    for n=1:g
        X=sort(datain(~isnan(datain(:,n)),n)); % clean & sort data
        if strfind(class(X),'int') %#ok<STRIFCND>
            X=single(X);
            m=single(m);
        end
        Xgi=X(X>=m(n));% X+: all values >= to median
        Xlj=X(X<=m(n));% X-: all values <= to median
        % compare all possible combinations
        Xi=repmat(Xgi ,1,length(Xlj)); % make HORZ matrix of X+  to match X-
        Xj=repmat(Xlj',length(Xgi),1); % make VERT matrix of X-' to match X+
        h=(((Xi-m(n))-(m(n)-Xj))./(Xi-Xj)); % formula 2.2
        ties=find(isnan(h));% only ties with median result in NaN values and are to be replaced
        
        if plotit
            figure(1234);clf;hold on;
            title(sprintf('median=%0.2f; ties=%d \nNumEl=%d (%2.2f%%)',m(n),length(ties),numel(h),100*length(ties)/numel(h)))
            colormap(jet); %blue/cyan <0; red/orange/yellow >0; green ==0;
            h(ties)=0; % just for plotting (green); to make sure rotations are correct for kernel
            tt=imagesc(double(h),[-1 1]); axis equal tight ij % proportions and orientation are critical
            set(gca,'yticklabel',Xgi(get(gca,'ytick')));
            set(gca,'xticklabel',Xlj(get(gca,'xtick')));
            xlabel('values <= to median')
            ylabel('values >= to median')
            drawnow
        end
        
        if any(ties)
            % ties always make a square (in upper right of plot) so use
            % flip_leftright (triangle_lower - triangle_upper)
            % to quickly create an appropriately sized kernel
            % -1 -1  0
            % -1  0  1
            %  0  1  1
            k=ones(sqrt(length(ties)));
            kernel=fliplr(tril(k)-triu(k));
            h(ties)=kernel; % formula 2.3
            
            if plotit
                set(tt,'CData',h)
                drawnow
            end
        end
        MC(n)=nanmedian(h(:));% formula 2.1
    end
