FROM node:22-alpine AS base
RUN apk add --no-cache libc6-compat
WORKDIR /app

# Install dependencies
COPY package.json pnpm-lock.yaml ./
RUN corepack enable && pnpm install --frozen-lockfile

# Build the application
FROM base AS build
WORKDIR /app

ENV NODE_ENV=production

COPY . .
RUN pnpm build

# Release image
FROM node:22-alpine AS release
WORKDIR /app

COPY --from=build /app/.next ./.next
COPY --from=build /app/public ./public
COPY --from=build /app/node_modules ./node_modules
COPY next.config.js package.json pnpm-lock.yaml ./

EXPOSE 3000

CMD ["pnpm", "start"]