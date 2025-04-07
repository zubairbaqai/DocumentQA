# Use official Node.js base image
FROM node:18

# Set working directory
WORKDIR /app

# Copy package files and install dependencies

COPY package.json package-lock.json ./

RUN npm install

# Copy the rest of the app
COPY . .

# Expose port
EXPOSE 8000

# Start the server
CMD ["node", "server.js"]
