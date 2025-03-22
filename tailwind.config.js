import forms from '@tailwindcss/forms'
import typography from '@tailwindcss/typography'
import colors from 'tailwindcss/colors'
import defaultTheme from 'tailwindcss/defaultTheme'

export default {
  mode: 'jit',
  content: ['./src/**/*.ts*'],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', ...defaultTheme.fontFamily.sans],
      },
      colors: {
        blue: colors.indigo,
        code: {
          green: '#a4f4c0',
          yellow: '#ffeb99',
          purple: '#d7a9ff',
          red: '#ff9999',
          blue: '#98dcff',
          white: '#ffffff',
        },
      },
      typography: (theme) => ({
        DEFAULT: {
          css: {
            color: theme('colors.gray.800'),
            a: {
              color: theme('colors.indigo.500'),
              '&:hover': {
                color: theme('colors.indigo.600'),
              },
              code: {color: theme('colors.indigo.400')},
            },
          },
        },
        dark: {
          css: {
            color: theme('colors.gray.100'),
            a: {
              color: theme('colors.indigo.300'),
              '&:hover': {
                color: theme('colors.indigo.200'),
              },
              code: {color: theme('colors.indigo.200')},
            },
            h1: {
              fontWeight: '700',
              letterSpacing: theme('letterSpacing.tight'),
              color: theme('colors.gray.50'),
            },
            h2: {
              fontWeight: '700',
              letterSpacing: theme('letterSpacing.tight'),
              color: theme('colors.gray.50'),
            },
            h3: {
              fontWeight: '600',
              color: theme('colors.gray.50'),
            },
            'h4,h5,h6': {
              color: theme('colors.gray.50'),
            },

            code: {
              backgroundColor: theme('colors.gray.700'),
            },

            hr: {borderColor: theme('colors.gray.600')},
            'ol li:before': {
              fontWeight: '600',
              color: theme('colors.gray.400'),
            },
            'ul li:before': {
              backgroundColor: theme('colors.gray.400'),
            },
            strong: {color: theme('colors.gray.100')},
            thead: {
              th: {
                color: theme('colors.gray.100'),
              },
            },
            tbody: {
              tr: {
                borderBottomColor: theme('colors.gray.600'),
              },
            },
            blockquote: {
              color: theme('colors.gray.100'),
              borderLeftColor: theme('colors.gray.600'),
            },
          },
        },
      }),
    },
  },
  variants: {
    typography: ['dark'],
  },
  plugins: [forms, typography],
}
