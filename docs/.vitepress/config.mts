import { defineConfig } from 'vitepress'

const baseConfig = {
  title: 'inference-learn',
  description: '从零用纯 C 写 Transformer 推理引擎',

  head: [
    ['link', { rel: 'icon', type: 'image/svg+xml', href: '/inference-learn/data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y=".9em" font-size="90">🧠</text></svg>' }],
    ['meta', { name: 'theme-color', content: '#5b8def' }]
  ],

  base: '/inference-learn/',
}

const zhThemeConfig = {
  siteTitle: 'inference-learn',
  socialLinks: [{ icon: 'github', link: 'https://github.com/jackiesre721/inference-learn' }],
  search: {
    provider: 'local',
    options: {
      translations: {
        button: { buttonText: '搜索文档', buttonAriaLabel: '搜索文档' },
        modal: { noResultsText: '无法找到相关结果', resetButtonTitle: '清除查询条件', footer: { selectText: '选择', navigateText: '切换' } }
      }
    }
  },
  sidebar: [
    { text: '📖 开始阅读', items: [{ text: '目录与阅读路径', link: '/cn/README' }] },
    { text: 'Part 1 · 基础', items: [
      { text: '第 1 章 基础概念', link: '/cn/01-basics' },
      { text: '第 2 章 权重的存储与加载', link: '/cn/02-weights' },
      { text: '第 3 章 JSON 解析器', link: '/cn/03-json' }
    ]},
    { text: 'Part 2 · 核心', items: [
      { text: '第 4 章 模型结构', link: '/cn/04-model' },
      { text: '第 5 章 前向传播 ★', link: '/cn/05-forward' },
      { text: '第 6 章 分词器 BPE', link: '/cn/06-tokenizer' },
      { text: '第 7 章 采样与生成', link: '/cn/07-sampling' }
    ]},
    { text: 'Part 3 · 进阶', items: [
      { text: '第 8 章 日志与可视化', link: '/cn/08-logging' },
      { text: '第 9 章 调试与验证方法论', link: '/cn/09-debugging' },
      { text: '第 10 章 生态对比与性能优化', link: '/cn/10-ecosystem' }
    ]},
    { text: '📚 附录', items: [
      { text: '附录 A 术语总表', link: '/cn/appendix-a-glossary' },
      { text: '附录 B 结构体速查', link: '/cn/appendix-b-structs' }
    ]}
  ],
  nav: [
    { text: '📖 目录', link: '/cn/README' },
    { text: '🏠 首页', link: '/cn/' },
    { text: '💻 源码', link: 'https://github.com/jackiesre721/inference-learn' },
    { text: '🌐 English', link: '/en/' }
  ],
  outline: { label: '本页目录', level: [2, 3] },
  docFooter: { prev: '上一章', next: '下一章' },
  footer: { message: 'MIT Licensed', copyright: 'inference-learn' },
  returnToTopLabel: '回到顶部',
  darkModeSwitchLabel: '主题',
  sidebarMenuLabel: '菜单'
}

const enThemeConfig = {
  siteTitle: 'inference-learn',
  socialLinks: [{ icon: 'github', link: 'https://github.com/jackiesre721/inference-learn' }],
  search: {
    provider: 'local',
    options: {
      translations: {
        button: { buttonText: 'Search', buttonAriaLabel: 'Search' },
        modal: { noResultsText: 'No results found', resetButtonTitle: 'Reset search', footer: { selectText: 'Select', navigateText: 'Navigate' } }
      }
    }
  },
  sidebar: [
    { text: '📖 Getting Started', items: [{ text: 'Table of Contents', link: '/en/README' }] },
    { text: 'Part 1 · Fundamentals', items: [
      { text: 'Ch.1 Basics', link: '/en/01-basics' },
      { text: 'Ch.2 Weights Loading', link: '/en/02-weights' },
      { text: 'Ch.3 JSON Parser', link: '/en/03-json' }
    ]},
    { text: 'Part 2 · Core', items: [
      { text: 'Ch.4 Model Architecture', link: '/en/04-model' },
      { text: 'Ch.5 Forward Pass ★', link: '/en/05-forward' },
      { text: 'Ch.6 Tokenizer BPE', link: '/en/06-tokenizer' },
      { text: 'Ch.7 Sampling & Generation', link: '/en/07-sampling' }
    ]},
    { text: 'Part 3 · Advanced', items: [
      { text: 'Ch.8 Logging & Visualization', link: '/en/08-logging' },
      { text: 'Ch.9 Debugging & Verification', link: '/en/09-debugging' },
      { text: 'Ch.10 Ecosystem Comparison', link: '/en/10-ecosystem' }
    ]},
    { text: '📚 Appendix', items: [
      { text: 'Appendix A Glossary', link: '/en/appendix-a-glossary' },
      { text: 'Appendix B Structs', link: '/en/appendix-b-structs' }
    ]}
  ],
  nav: [
    { text: '📖 Contents', link: '/en/README' },
    { text: '🏠 Home', link: '/en/' },
    { text: '💻 Source', link: 'https://github.com/jackiesre721/inference-learn' },
    { text: '🌐 中文', link: '/cn/' }
  ],
  outline: { label: 'On This Page', level: [2, 3] },
  docFooter: { prev: 'Previous', next: 'Next' },
  footer: { message: 'MIT Licensed', copyright: 'inference-learn' },
  returnToTopLabel: 'Back to top',
  darkModeSwitchLabel: 'Appearance',
  sidebarMenuLabel: 'Menu'
}

export default defineConfig({
  ...baseConfig,
  locales: {
    root: { label: 'root', lang: 'en' },
    'cn': { label: '简体中文', lang: 'zh-CN', themeConfig: zhThemeConfig },
    'en': { label: 'English', lang: 'en-US', themeConfig: enThemeConfig }
  },
  markdown: {
    lineNumbers: true,
    theme: { light: 'github-light', dark: 'github-dark' }
  }
})
