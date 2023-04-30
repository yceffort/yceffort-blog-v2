module.exports = {
  "extends": [
    "plugin:@next/next/recommended",
    "@titicaca/eslint-config-triple",
    "@titicaca/eslint-config-triple/prettier"
  ],
  rules: {
    'react/react-in-jsx-scope': 'off',
    '@typescript-eslint/no-explicit-any': 'warn',
    '@typescript-eslint/explicit-member-accessibility': 'off',
    'jsx-a11y/anchor-has-content': 'off',
    'jsx-a11y/accessible-emoji': 'off',
    'jsx-a11y/anchor-is-valid': 'off',
  },
  parserOptions: {
    requireConfigFile: false,
  }
}
