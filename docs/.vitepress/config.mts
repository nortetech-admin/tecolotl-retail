import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'Tecolotl Retail Docs',
  description: 'Retail computer vision documentation',

  themeConfig: {
    nav: [
      { text: 'Getting Started', link: '/getting-started/' },
      { text: 'Architecture', link: '/architecture/' },
      { text: 'Hardware', link: '/hardware/raspberry-pi' },
      { text: 'Vision', link: '/vision/people-detection' }
    ],

    sidebar: [
      {
        text: 'Getting Started',
        items: [
          { text: 'Overview', link: '/getting-started/' }
        ]
      },
      {
        text: 'Architecture',
        items: [
          { text: 'System Architecture', link: '/architecture/' }
        ]
      },
      {
        text: 'Hardware',
        items: [
          { text: 'Raspberry Pi', link: '/hardware/raspberry-pi' },
          { text: 'IMX500', link: '/hardware/imx500' },
          { text: 'Display Pipeline', link: '/hardware/display-pipeline' }
        ]
      },
      {
        text: 'Vision',
        items: [
          { text: 'People Detection', link: '/vision/people-detection' },
          { text: 'Keypoints', link: '/vision/keypoints' },
          { text: 'Tracking', link: '/vision/tracking' },
          { text: 'Shelf Attention', link: '/vision/shelf-attention' }
        ]
      },
      {
        text: 'Reference',
        items: [
          { text: 'Reference', link: '/reference/' }
        ]
      }
    ]
  }
})