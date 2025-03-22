import naverpay from '@naverpay/eslint-config'
import naverpayPlugin from '@naverpay/eslint-plugin'
import {defineConfig} from 'eslint/config'

export default defineConfig([
  {
    ignores: ['**/.next/*'],
  },
  ...naverpay.configs.react,
  ...naverpay.configs.packageJson,
  {
    plugins: {
      '@naverpay': naverpayPlugin,
    },
    rules: {
      'jsx-a11y/anchor-has-content': 'off',
      'jsx-a11y/accessible-emoji': 'off',
      'jsx-a11y/anchor-is-valid': 'off',
    },
  },
])
