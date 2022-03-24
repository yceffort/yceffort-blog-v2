module.exports = {
  plugins: [
    ...require('@titicaca/eslint-config-triple/prettierrc'),
    require('prettier-plugin-tailwindcss'),
  ],
  tailwindConfig: './tailwind.config.js',
}
