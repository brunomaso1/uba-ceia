# Usar imagen base de Nginx
FROM nginx:alpine

# Remueve la configuración default
RUN rm /etc/nginx/conf.d/default.conf

# Copia tu configuración personalizada
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copiar archivos locales al directorio HTML de Nginx
COPY . /usr/share/nginx/html

# Exponer el puerto 80 (HTTP)
EXPOSE 80