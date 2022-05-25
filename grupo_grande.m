
clearvars,clc
close all

txt = ["Seleccione un archivo de audio","Seleccione la topología del filtro que desea aplicar:\n1. FIR por enventanado" + ...
    "\n2. FIR por muestreo en frecuencia\n3. FIR por Parks-McClellan\n4. IIR Butterworth\n5. IIR Chebyshov I\n6. IIR Chebyshov II" + ...
    "\n7. IIR Elíptico\n: ","Seleccione el tipo de ventana:\n1. Rectangular\n2. Hann\n3. Hamming\n4. Kaiser\n: ","Seleccione el tipo de filtro:" + ...
    "\n1. Pasa bajas\n2. Pasa altas\n3. Pasa bandas\n4. Rechaza bandas\n: ","Ingrese la frecuencia de corte: "...
    "Ingrese la frecuencia de corte inferior: ","Ingrese la frecuencia de corte superior: "];

aud = uigetfile({'*.*','All Files (*.*)'},txt(1));
[y, Fs] = audioread(aud);
Ts = 1/Fs; % Periodo de muestreo
Ndat = 0:length(y)-1; % Cantidad de datos
t = Ndat*Ts; 
y = (y(:,1)+y(:,2))/2; % Audio convertido a monofónico

top_filter = input(txt(2));

% Sección del filtro FIR enventanado 
ord_FIR = 10;
ord_IIR = 5;
if top_filter == 1 
    wtype = input(txt(3));
    if wtype == 1
        wtype = rectwin(ord_FIR+1);    % Ventana rectangular
    elseif wtype == 2
        wtype = hann(ord_FIR+1);       % Ventana Hanning
    elseif wtype == 3
        wtype = hamming(ord_FIR+1);    % Ventana Hamming
    elseif wtype == 4           
        wtype = kaiser(ord_FIR+1,0.7); % Ventana Kaiser
    end
end

% Tipo de filtro
fil_type = input(txt(4));

if fil_type == 1
    fc = input(txt(5));
    Fc = fc/(Fs/2);
    fil_type = 'low';
    f = [0 Fc Fc+0.001 1];
    Ak = [1 1 0 0];
elseif fil_type == 2
    fc = input(txt(5));
    Fc = fc/(Fs/2);
    fil_type = 'high';
    f = [0 Fc Fc+0.001 1];
    Ak = [0 0 1 1];
elseif fil_type == 3
    fc1 = input(txt(6));
    fc2 = input(txt(7));
    fc = [fc1 fc2];
    Fc = fc./(Fs/2);
    fil_type = 'bandpass';
    f = [0 Fc(1) Fc(1)+0.001 Fc(2) Fc(2)+0.001 1];
    Ak = [0 0 1 1 0 0];
elseif fil_type == 4
    fc1 = input(txt(6));
    fc2 = input(txt(7));
    fc = [fc1 fc2];
    Fc = fc./(Fs/2);
    fil_type = 'stop';
    f = [0 Fc(1) Fc(1)+0.001 Fc(2) Fc(2)+0.001 1];
    Ak = [1 1 0 0 1 1];
end

% Tipo de filtrado para cada filtro FIR
if top_filter == 1     % Enventando
    a = 1;
    b = fir1(ord_FIR,Fc,fil_type,wtype);
elseif top_filter == 2 % Muestreo en frecuencia 
    a = 1;
    b = fir2(ord_FIR,f,Ak);
elseif top_filter == 3 % Parks-McClellan
    a = 1;
    b = firpm(ord_FIR,f,Ak)
end

% Filtros IIR 
if top_filter == 4 % Filtro Butterworth 
    [num,den] = butter(ord_IIR,2*pi*fc,fil_type,'s');
    [numd,dend] = bilinear(num,den,Fs); % Bilinear
    b = numd;
    a = dend;
elseif top_filter == 5 % Filtro Chevyshov I
    Rp = 0.5;
    [num,den] = cheby1(ord_IIR,Rp,2*pi*fc,fil_type,'s');
    [numd,dend] = bilinear(num,den,Fs);
    b = numd;
    a = dend;
elseif top_filter == 6
    Rs = 35; % Filtro Chevyshov II
    [num,den] = cheby2(ord_IIR,Rs,2*pi*fc,fil_type,'s');
    [numd,dend] = bilinear(num,den,Fs); 
    b = numd;
    a = dend;
elseif top_filter == 7 % Elíptico
    Rp = 0.5;
    Rs = 35;
    [num,den] = ellip(ord_IIR,Rp,Rs,2*pi*fc,fil_type,'s');
    [numd,dend] = bilinear(num,den,Fs);
    b = numd;
    a = dend;
end

% Gráfica respuesta en frecuencia de magnitud
[H, w] = freqz(b,a,Fs); 

figure
subplot 211
plot(Fs*w/(2*pi),20*log10(abs(H))) % En funcion de f [Hz]
xlabel 'Frecuencia [Hz]',ylabel 'Magnitud [dB]',axis tight, 
title 'Respuesta en frecuencia de magnitud'

subplot 212
plot(Fs*w/(2*pi),angle(H)) % En funcion de f [Hz]
xlabel 'Frecuencia [Hz]',ylabel 'Fase [rad]',axis tight, 
title 'Respuesta en frecuencia de fase'

% Gráfica de la señal de audio original y filtrada, en el dominio del tiempo
figure
filt_sig = filter(b,a,y);
plot(t,y,'b',t,filt_sig,'r' )
xlabel 'Tiempo [s]', ylabel 'Amplitud [mV]',axis tight 
title 'Señal de audio'
legend('Audio original','Audio filtrado')

% ESD 
figure
w = linspace(0,2*pi,length(y));
ESD_y = abs(fft(y)).^2; % ESD señal original
w = w(1:length(ESD_y)/2);
ESD_fsig = abs(fft(filt_sig)).^2; % ESD señal filtrada
plot(Fs*w/(2*pi),ESD_y(1:length(ESD_y)/2),'b',Fs*w/(2*pi),ESD_fsig(1:length(ESD_fsig)/2),'r')
xlabel 'Frecuencia [Hz]', ylabel 'ESD', axis tight 
title 'ESD (Densidad espectral de energía)'
legend('Audio original','Audio filtrado')

% STFT señal original
figure
m = 40e-3; % Tiempo lento [s]
M = m*Fs;  % Tamaño de la ventana [muestras]
N = length(y);
R = M/2; % Desplazamiento [muestras]
N_m = floor((N-M)/R+1); % Cantidad total de ventanas
Wintype = rectwin(M);   % Ventana rectangular de tamaño M
STFT = zeros(M,N_m);   
STFT(:,1) = abs(fft(y(1:M).*Wintype)).^2;  % Primer ventana

for i = 1:N_m-1 % Contador para recorrer las columnas (desde la segunda)
    STFT(:,i+1) = abs(fft(y(i*R:(i*R+M)-1).*Wintype)).^2;
end

Ndat1 = 0:R:length(y)-1; % Cantidad de datos
t1 = Ndat1*Ts;
w1 = linspace(0,2*pi,M);
w1 = w1(1:M/2);

% Espectrograma señal original
subplot 211
mesh(t1(1:end-2),Fs*w1/(2*pi),STFT(1:M/2,:))
colorbar
xlabel 'Tiempo [s]', ylabel 'Frecuencia [Hz]',zlabel 'STFT'
title 'Espectrograma de la señal original'

% STFT señal filtrada
STFT_f = zeros(M,N_m);   
STFT_f(:,1) = abs(fft(filt_sig(1:M).*Wintype)).^2; 

for i = 1:N_m-1 
    STFT_f(:,i+1) = abs(fft(filt_sig(i*R:(i*R+M)-1).*Wintype)).^2;
end

% Espectrograma señal filtrada
subplot 212
mesh(t1(1:end-2),Fs*w1/(2*pi),STFT_f(1:M/2,:))
colorbar
xlabel 'Tiempo [s]', ylabel 'Frecuencia [Hz]',zlabel 'STFT'
title 'Espectrograma de la señal filtrada'