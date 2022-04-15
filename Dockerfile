FROM node:16-alpine AS base
RUN apk add --no-cache libc6-compat
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

# build
FROM base AS build
WORKDIR /app

ENV NODE_ENV=production

COPY . .
RUN npm run build

# release
FROM node:16-alpine as release
WORKDIR /app

COPY --from=build /app/.next ./.next
COPY --from=build /app/public ./public
COPY --from=build /app/node_modules ./node_modules
COPY next.config.js package.json package-lock.json ./

EXPOSE 3000

CMD ["npm", "start"]