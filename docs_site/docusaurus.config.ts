import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'GradCore-Tensor',
  tagline: 'A Lightweight C++ Neural Network & Autograd Library',

  favicon: 'img/logo_rm.svg',
  
  url: 'https://spandan11106.github.io', 
  baseUrl: '/GradCore-Tensor/', 

  organizationName: 'spandan11106', // Your GitHub org/user name.
  projectName: 'GradCore-Tensor', // Your repo name.
  deploymentBranch: 'gh-pages',
  trailingSlash: false,

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          // editUrl: 'https://github.com/spandan11106/GradCore-Tensor/tree/main/docs_site/',
        },
        blog: {
          showReadingTime: true,
          // editUrl: 'https://github.com/spandan11106/GradCore-Tensor/tree/main/docs_site/',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/docusaurus-social-card.jpg',
    navbar: {
      title: 'GradientCore',
      logo: {
        alt: 'GradientCore Logo',
        src: 'img/logo_rm.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Docs',
        },
        {to: '/blog', label: 'Blog', position: 'left'},
        {
          href: 'https://github.com/spandan11106/GradCore-Tensor',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {
              label: 'Getting Started',
              to: '/docs/intro'
            },
            {
              label: 'Tutorials',
              to: '/docs/tutorials/tutorial-1-california-housing',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'Blog',
              to: '/blog',
            },
            {
              label: 'GitHub',
              href: 'https://github.com/spandan11106/GradCore-Tensor',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} GradientCore. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
