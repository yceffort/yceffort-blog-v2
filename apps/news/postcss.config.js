import babelConfig from './.babelrc.json' with {type: 'json'}

const config = {
  plugins: {
    '@stylexjs/postcss-plugin': {
      include: ['src/**/*.{js,jsx,ts,tsx}'],
      babelConfig: {
        babelrc: false,
        configFile: false,
        parserOpts: {plugins: ['typescript', 'jsx']},
        plugins: babelConfig.plugins,
      },
      useCSSLayers: false,
    },
  },
}
export default config
