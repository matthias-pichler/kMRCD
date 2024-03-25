% script to show different ways to use adjusted_boxplot
% with built in data sets
%%
clear
load examgrades.mat
XTickLabels={'test A','test B','test C','test D','test E'};
[~,ind]=sort(grades(:,1)); % sort all grades based on first test
grades=grades(ind,:); % this will auto shift outlyers from left(low) to right(high)
groupcount=size(grades,2);
colors=lines(groupcount);
cc=mat2cell(colors,ones(groupcount,1),3);
figure(4321);clf

subplot(2,2,1);hold on;title('whisker_boxplot')
whisker_boxplot(1:groupcount,grades, color=cc);
set(gca,'XTick',1:length(XTickLabels),'XTickLabels',XTickLabels)

subplot(2,2,2);hold on;title('adjusted_boxplot')
adjusted_boxplot(1:groupcount,grades, color=cc);
set(gca,'XTick',1:length(XTickLabels),'XTickLabels',XTickLabels)

subplot(2,2,3);hold on;title(sprintf('whisker_boxplot \nwith bubbles & saturation set to white'))
whisker_boxplot(1:groupcount,grades, color=cc, bub=true, sat=0);
set(gca,'XTick',1:length(XTickLabels),'XTickLabels',XTickLabels)

subplot(2,2,4);hold on;title(sprintf('adjusted_boxplot \nwith bubbles & saturation set to white'))
adjusted_boxplot(1:groupcount,grades, color=cc, bub=true, sat=0);
set(gca,'XTick',1:length(XTickLabels),'XTickLabels',XTickLabels)

%%
clear
load discrim.mat
XTickLabels=strtrim(mat2cell(categories,ones(size(categories,1),1),size(categories,2)))';
namecount=size(names,1);
cc=strtrim(mat2cell(names,ones(namecount,1),43));
C=cellfun(@(x) strsplit(x,', '),cc,'uni',0);
simplestate=cellfun(@(x) upper(x{end}(1:2)),C,'uni',0);
[ustate,order]=unique(sort(simplestate));
statecount=diff([order; length(simplestate)+1]);
bigstates=ustate(statecount>15); %find only states with >15 scores/cities
runthese=ismember(simplestate,bigstates);
xs=.4/3; %the best half width for a 'group or catagory' is .4 and there are 3 big states
width=xs*.85;% width of each box with a small gap between subgroups
sgxs=[-xs 0 xs]*2;% SubGroups X Shift (twice the half width)
cc={[0 0 1],[1 .5 0],[1 0 0]};% subgroups color (blue orange red)
figure(2314);clf
subplot(2,1,1);hold on;title('whisker_boxplot')
for ii=1:length(bigstates)
    for jj=1:length(XTickLabels)
        ratings(ismember(simplestate,bigstates(ii)),jj);
        h1{ii}=whisker_boxplot(jj+sgxs(ii),ratings(ismember(simplestate,bigstates(ii)),jj), color={cc{ii}}, width=width);
    end
end
set(gca,'XTick',1:length(XTickLabels),'XTickLabels',XTickLabels)
legend([h1{1}(1) h1{2}(1) h1{3}(1)],bigstates)

subplot(2,1,2);hold on;title('adjusted_boxplot')
for ii=1:length(bigstates)
    for jj=1:length(XTickLabels)
        ratings(ismember(simplestate,bigstates(ii)),jj);
        h1{ii}=adjusted_boxplot(jj+sgxs(ii),ratings(ismember(simplestate,bigstates(ii)),jj), color={cc{ii}}, width=width);
    end
end
set(gca,'XTick',1:length(XTickLabels),'XTickLabels',XTickLabels)
legend([h1{1}(1) h1{2}(1) h1{3}(1)],bigstates)


%%
clear
load hospital
namecount=size(hospital,1);
shiftxs=hospital.Age-min(hospital.Age);
shiftxs=shiftxs/max(shiftxs)*2-1; % set range for shifting each point to -1:1 based on age
Sex=strtrim(mat2cell(char(hospital.Sex),ones(namecount,1),6));
male=strcmpi(Sex,'male');

xs=.4/2; %the best half width for a 'group or catagory' is .4 and there are 2 subgroups (male/female)
width=xs*.85;% width of each box with a small gap between subgroups

%
% plot weight, then plot pressure(1), then pressure(2)
XTickLabels={'weight' 'systolic' 'diastolic'};
figure(8134);clf;
subplot(2,2,1);hold on;title('whisker_boxplot')
l1=whisker_boxplot(1-xs,hospital.Weight( male), color={[0 0 1]}, shiftxs=shiftxs*width, width=width);
l2=whisker_boxplot(1+xs,hospital.Weight(~male), color={[1 0 0]}, shiftxs=shiftxs*width, width=width);

whisker_boxplot(2-xs,hospital.BloodPressure( male,1), color={[0 0 1]}, shiftxs=shiftxs*width,width=width);
whisker_boxplot(2+xs,hospital.BloodPressure(~male,1), color={[1 0 0]}, shiftxs=shiftxs*width,width=width);

whisker_boxplot(3-xs,hospital.BloodPressure( male,2), color={[0 0 1]}, shiftxs=shiftxs*width,width=width);
whisker_boxplot(3+xs,hospital.BloodPressure(~male,2), color={[1 0 0]}, shiftxs=shiftxs*width,width=width);
set(gca,'XTick',1:length(XTickLabels),'XTickLabels',XTickLabels)
legend([l1(1) l2(1)],{'male','female'});

subplot(2,2,2);hold on;title('adjusted_boxplot')
adjusted_boxplot(1-xs,hospital.Weight( male), color={[0 0 1]}, shiftxs=shiftxs*width, width=width);
adjusted_boxplot(1+xs,hospital.Weight(~male), color={[1 0 0]}, shiftxs=shiftxs*width, width=width);

adjusted_boxplot(2-xs,hospital.BloodPressure( male,1), color={[0 0 1]}, shiftxs=shiftxs*width, width=width);
adjusted_boxplot(2+xs,hospital.BloodPressure(~male,1), color={[1 0 0]}, shiftxs=shiftxs*width, width=width);

adjusted_boxplot(3-xs,hospital.BloodPressure( male,2), color={[0 0 1]}, shiftxs=shiftxs*width, width=width);
adjusted_boxplot(3+xs,hospital.BloodPressure(~male,2), color={[1 0 0]}, shiftxs=shiftxs*width, width=width);
set(gca,'XTick',1:length(XTickLabels),'XTickLabels',XTickLabels)

subplot(2,2,3);hold on;title('whisker_boxplot, with bubbles & saturation set to white')
whisker_boxplot(1-xs,hospital.Weight( male), color={[0 0 1]}, shiftxs=shiftxs*width,width=width, bub=true, sat=0);
whisker_boxplot(1+xs,hospital.Weight(~male), color={[1 0 0]}, shiftxs=shiftxs*width,width=width, bub=true, sat=0);

whisker_boxplot(2-xs,hospital.BloodPressure( male,1), color={[0 0 1]}, shiftxs=shiftxs*width,width=width, bub=true, sat=0);
whisker_boxplot(2+xs,hospital.BloodPressure(~male,1), color={[1 0 0]}, shiftxs=shiftxs*width,width=width, bub=true, sat=0);

whisker_boxplot(3-xs,hospital.BloodPressure( male,2), color={[0 0 1]}, shiftxs=shiftxs*width,width=width, bub=true, sat=0);
whisker_boxplot(3+xs,hospital.BloodPressure(~male,2), color={[1 0 0]}, shiftxs=shiftxs*width,width=width, bub=true, sat=0);
set(gca,'XTick',1:length(XTickLabels),'XTickLabels',XTickLabels)

subplot(2,2,4);hold on;title('adjusted_boxplot, with bubbles & saturation set to white')
adjusted_boxplot(1-xs,hospital.Weight( male), color={[0 0 1]}, shiftxs=shiftxs*width, width=width, bub=true, sat=0);
adjusted_boxplot(1+xs,hospital.Weight(~male), color={[1 0 0]}, shiftxs=shiftxs*width, width=width, bub=true, sat=0);

adjusted_boxplot(2-xs,hospital.BloodPressure( male,1), color={[0 0 1]}, shiftxs=shiftxs*width, width=width, bub=true, sat=0);
adjusted_boxplot(2+xs,hospital.BloodPressure(~male,1), color={[1 0 0]}, shiftxs=shiftxs*width, width=width, bub=true, sat=0);

adjusted_boxplot(3-xs,hospital.BloodPressure( male,2), color={[0 0 1]}, shiftxs=shiftxs*width, width=width, bub=true, sat=0);
adjusted_boxplot(3+xs,hospital.BloodPressure(~male,2), color={[1 0 0]}, shiftxs=shiftxs*width, width=width, bub=true, sat=0);
set(gca,'XTick',1:length(XTickLabels),'XTickLabels',XTickLabels)
