const createConfig = require('@titicaca/eslint-config-triple/create-config')

const { extends: extendConfigs, overrides } = createConfig({
  type: 'frontend',
  project: './tsconfig.json',
  allowedNames: [
    'cloud_name',
    'api_key',
    'api_secret',
    'access_type',
    'public_id',
    'unique_filename',
    'client_email',
    'private_key',
    'project_id',
    'event_category',
    'event_label',
    'non_interaction',
  ],
})

module.exports = {
  extends: [...extendConfigs, 'plugin:@next/next/recommended'],
  overrides,
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
  },
}
