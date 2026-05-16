import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import CodeBlock from '@theme/CodeBlock';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className={styles.heroTitle}>
          {siteConfig.title}
        </Heading>
        <p className={styles.heroSubtitle}>{siteConfig.tagline}</p>
        <p className={styles.heroDescription}>
          An open-source, zero-allocation C++17 deep learning framework. 
          Built for bare-metal execution speeds, dynamic computation graphs, 
          and ultimate transparency in machine learning education.
        </p>
        <div className={styles.buttons}>
          <Link
            className="button button--primary button--lg"
            to="/docs/intro">
            GET STARTED
          </Link>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro"> {/* Link this to your tutorials page later */}
            TUTORIALS
          </Link>
          <Link
            className="button button--secondary button--lg"
            href="https://github.com/spandan11106/GradCore-Tensor">
            GITHUB
          </Link>
        </div>
      </div>
    </header>
  );
}

function QuickStartCode() {
  const codeSnippet = `
#include <gradient.hpp>

using namespace gradientcore;

int main() {
    // Zero-overhead memory arenas
    Arena *perm_arena = Arena::create(MiB(16), MiB(1), false);
    Arena *graph_arena = Arena::create(MiB(64), MiB(1), true);

    // PyTorch-like Model Definition
    nn::Model model(perm_arena, graph_arena);
    model.add_layer(new (perm_arena->push<nn::Linear>()) nn::Linear(perm_arena, 8, 128));
    model.add_layer(new (perm_arena->push<nn::ReLU>()) nn::ReLU());
    model.add_layer(new (perm_arena->push<nn::Linear>()) nn::Linear(perm_arena, 128, 1));

    // Compile & Train!
    model.compile(nn::OptimizerType::ADAMW, nn::LossType::HUBER, 0.001f, 200, 128);
    model.train(X_train, Y_train);

    return 0;
}`;

  return (
    <section className={styles.codeSection}>
      <div className="container">
        <div className={styles.codeSectionInner}>
          <Heading as="h2">Build Fast. Execute Faster.</Heading>
          <CodeBlock language="cpp" title="train.cpp">
            {codeSnippet}
          </CodeBlock>
        </div>
      </div>
    </section>
  );
}

export default function Home(): JSX.Element {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Home | ${siteConfig.title}`}
      description="High-Performance C++17 Deep Learning Framework">
      <HomepageHeader />
      <main>
        <QuickStartCode />
      </main>
    </Layout>
  );
}
