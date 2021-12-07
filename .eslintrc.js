const { extendedRules } = require('./naming-convention')

module.exports = {
  extends: [
    'eslint-config-yceffort/typescript',
    'plugin:@next/next/recommended',
  ],
  rules: {
    '@typescript-eslint/naming-convention': extendedRules,
    'react/react-in-jsx-scope': 'off',
  },
}
