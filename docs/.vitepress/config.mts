import { defineConfig } from 'vitepress'

export default defineConfig({
  lang: 'zh-CN',
  title: 'inference-learn',
  description: '从零用纯 C 写 Transformer 推理引擎，理解 LLM 的每一个零件',

  // GitHub 仓库关联
  repo: 'jackiesre721/inference-learn',
  repoLink: 'https://github.com/jackiesre721/inference-learn',

  // 部署到 GitHub Pages 时的 base 路径
  base: '/inference-learn/',

  // 主题配置
  themeConfig: {
    // 站点标题（左上角）
    siteTitle: 'inference-learn',

    // GitHub 仓库链接（右上角）
    socialLinks: [
      { icon: 'github', link: 'https://github.com/jackiesre721/inference-learn' }
    ],

    // 全文搜索
    search: {
      provider: 'local',
      options: {
        translations: {
          button: {
            buttonText: '搜索文档',
            buttonAriaLabel: '搜索文档'
          },
          modal: {
            noResultsText: '无法找到相关结果',
            resetButtonTitle: '清除查询条件',
            footer: {
              selectText: '选择',
              navigateText: '切换'
            }
          }
        }
      }
    },

    // 侧边栏目录
    sidebar: [
      {
        text: '📖 开始阅读',
        items: [
          { text: '目录与阅读路径', link: '/README' }
        ]
      },
      {
        text: ' Part 1 · 基础',
        items: [
          { text: '第 1 章 基础概念', link: '/01-basics' },
          { text: '第 2 章 权重的存储与加载', link: '/02-weights' },
          { text: '第 3 章 JSON 解析器', link: '/03-json' }
        ]
      },
      {
        text: ' Part 2 · 核心',
        items: [
          { text: '第 4 章 模型结构', link: '/04-model' },
          { text: '第 5 章 前向传播 ★', link: '/05-forward' },
          { text: '第 6 章 分词器 BPE', link: '/06-tokenizer' },
          { text: '第 7 章 采样与生成', link: '/07-sampling' }
        ]
      },
      {
        text: ' Part 3 · 进阶',
        items: [
          { text: '第 8 章 日志与可视化', link: '/08-logging' },
          { text: '第 9 章 调试与验证方法论', link: '/09-debugging' },
          { text: '第 10 章 生态对比与性能优化', link: '/10-ecosystem' }
        ]
      },
      {
        text: '📚 附录',
        items: [
          { text: '附录 A 术语总表', link: '/appendix-a-glossary' },
          { text: '附录 B 结构体速查', link: '/appendix-b-structs' }
        ]
      }
    ],

    // 顶部导航栏
    nav: [
      { text: '📖 文档', link: '/README' },
      { text: '🏠 首页', link: 'https://github.com/jackiesre721/inference-learn' },
      { text: '💻 源码', link: 'https://github.com/jackiesre721/inference-learn' }
    ],

    // 上一页/下一页
    outline: {
      label: '本页目录',
      level: [2, 3]
    },

    docFooter: {
      prev: '上一章',
      next: '下一章'
    },

    // 页脚
    footer: {
      message: 'MIT Licensed',
      copyright: 'inference-learn · 从零理解 LLM 推理引擎'
    },

    // 页面滚动到顶部按钮
    returnToTopLabel: '回到顶部',

    // 暗色模式切换
    darkModeSwitchLabel: '主题',
    sidebarMenuLabel: '菜单',
    darkModeSwitchTitle: '切换深色模式',
    lightModeSwitchTitle: '切换浅色模式'
  },

  // Markdown 配置
  markdown: {
    // 行高亮
    lineNumbers: true,
    // 主题
    theme: { light: 'github-light', dark: 'github-dark' }
  }
})
