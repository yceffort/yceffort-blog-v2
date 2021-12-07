const createConfig = require('@titicaca/eslint-config-triple/create-config')

const { extends: extendConfigs, overrides } = createConfig({ type: 'frontend', project: './tsconfig.json' })

module.exports = {
  extends: [
    ...extendConfigs,
    'plugin:@next/next/recommended',
  ],
  overrides,
  rules: {
    'react/react-in-jsx-scope': 'off',
  },
  parserOptions: {
    requireConfigFile: false,
  },
}
