clc
clear all
close all
addpath ./Utilities
set(0,'defaulttextinterpreter','latex')
R = [216, 82,  24 ]/255; R = R/norm(R);
B = [0 , 113, 188]/255;  B = B/norm(B);
MS = 'MarkerSize';

%% Hindcast SST
load('SSTData/DailySatData')
load('SSTData/XStarMap')
load('SSTData/PredictionMap')
imax = 81; jmax = 81; Np  = imax*jmax;
X = XStar(1:Np,1); X = reshape(X,imax,jmax);
Y = XStar(1:Np,2); Y = reshape(Y,imax,jmax);

xmin = min(min(X));
xmax = max(max(X));
ymin = min(min(Y));
ymax = max(max(Y));


stateName = 'Massachusetts';
states = geoshape(shaperead('usastatehi', 'UseGeoCoords', true));
ma = states(strcmp(states.Name, stateName));
Mass_iden = inpolygon(X(:),Y(:),ma.Longitude,ma.Latitude);

for Day=[1  120  240 ]
    fig = figure;
    set(fig,'units','normalized','outerposition',[0 0 .9 .6])
    colormap(jet)
    
    SST_mean = mean_star((Day-1)*Np+1:Day*Np);
    SST_mean(find(Mass_iden==1))=nan;
    SST_mean = reshape(SST_mean,imax, jmax);
    
    SST_std = std_star((Day-1)*Np+1:Day*Np);
    SST_std(find(Mass_iden==1))=nan;
    SST_std = reshape(SST_std,imax, jmax);
    
    %% --------------------------------------------------------------------
    subplot(1,3,1)
    surf(Data.X,Data.Y,Data.SST(:,:,Day)); view([0 0 1]); axis equal; axis tight; hold on; colorbar
    xlabel('Longitude'); ylabel('Latitude')
    plot(ma.Longitude,ma.Latitude,'LineWidth',2);
    axis([xmin xmax ymin ymax])
    set(gca,'FontSize',25);
    set(gca,'xtick',[])
    set(gca,'ytick',[])
    set(gca,'yticklabel',{[]})
    set(gca,'xticklabel',{[]})
    set(gcf, 'Color', 'w');
    grid
    cl = caxis;
    title('Data: Satellite (MODIS)')
    
    %% --------------------------------------------------------------------
    subplot(1,3,2)
    contourf(X,Y,SST_mean,20); view([0 0 1]); axis equal; axis tight; hold on;colorbar
    xlabel('Longitude'); ylabel('Latitude')
    plot(ma.Longitude,ma.Latitude,'LineWidth',2); hold off
    axis([xmin xmax ymin ymax])
    set(gca,'FontSize',25);
    set(gca,'xtick',[])
    set(gca,'ytick',[])
    set(gca,'yticklabel',{[]})
    set(gca,'xticklabel',{[]})
    set(gcf, 'Color', 'w');
    grid
    caxis(cl)
    title('PGP Model: Mean')
    
    %% --------------------------------------------------------------------
    subplot(1,3,3)
    contourf(X,Y,SST_std,20);  axis equal; axis tight; hold on;colorbar
    xlabel('Longitude'); ylabel('Latitude')
    plot(ma.Longitude,ma.Latitude,'LineWidth',2); hold off
    axis([xmin xmax ymin ymax])
    set(gca,'FontSize',25);
    set(gca,'xtick',[])
    set(gca,'ytick',[])
    set(gca,'yticklabel',{[]})
    set(gca,'xticklabel',{[]})
    set(gcf, 'Color', 'w');
    cl = caxis; cl(2)= 0.5*cl(2);
    caxis(cl)
    title('PGP Model: Uncertainty Map')
    
    %% --------------------------------------------------------------------
    filename = ['Fig/SST-Day',num2str(Day)];
    set(gcf, 'InvertHardCopy', 'off');
    print(fig,filename,'-depsc')   
end

%% Independent Valiation with Buoy A01
load('SSTData/XStarBuoy')
load('SSTData/PredictionBuoy')
t = XStar(:,3)- XStar(1,3);

fig = figure
plot(t, YStar,'o','color',R,MS,6); hold on
plot(t,mean_star,'LineWidth',3,'color',B);
boundedline(t, mean_star, 2.0*std_star, ':', 'alpha','cmap', B);
xlabel('Days')
ylabel('SST ($^{\circ}$ C)')
xlim([1 365])
ylim([-5 25])
axis square
set(gca,'FontSize',15);
set(gcf, 'Color', 'w');
grid

hl = legend('NERACOOS Buoy A01','Parametric GP','Two standard deviation','Location','southeast')
legend boxoff
set(hl,'Interpreter','latex')

filename = 'Fig/SST-Buoy';
set(gcf, 'InvertHardCopy', 'off');
print(fig,filename,'-depsc')

