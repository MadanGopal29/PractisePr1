# Use a lightweight base image
FROM nginx:alpine

# Remove the default Nginx configuration file
RUN rm /etc/nginx/conf.d/default.conf

# Copy over the custom Nginx configuration file
COPY nginx.conf /etc/nginx/conf.d/

# Copy the HTML file(s) into the container
COPY index.html /usr/share/nginx/html/

# Expose port  6910
EXPOSE  6910

# Command to run Nginx
CMD ["nginx", "-g", "daemon off;"]

