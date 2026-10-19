---
title: "If the Code Meets Spec and the Bugs Get Fixed, Does It Matter That <em>You Can't Read It</em>?"
tags:
  - ai
  - essay
  - career
  - code-review
  - software-engineering
published: true
date: 2026-06-12
description: "If the spec is satisfied and the bugs get fixed, do you still need to be able to read the code? Understanding doesn't disappear: this essay traces where it moves."
series: 'Judgment in the AI Era'
seriesOrder: 1
art:
  undraw: books
  layout: glyph
  hue: violet
  tone: dark
  hero: '숨겨진 커플링'
---

## Overview

A PR goes up. An agent leaves the first-pass review, CI is green, and the product manager approves. The person who actually merged the code never read it all the way through. It's a scene that keeps getting more common.

Two questions branch off from here. The first is verification. To judge whether the AI's review is correct, whether the code really satisfies the spec, someone, in the end, has to read the code. To catch the places where AI is confidently wrong, you need to know not just the code but how the model fails. Not long ago I handed an agent a bug that only fired in one specific environment. It worked its way down a list of the usual suspects, all wrong, and never once pointed at the single conditional that was the actual culprit. It got caught only after I read the code myself. The second question is growth. In an era where AI takes over writing the code, what exactly is someone just starting out supposed to get good at? Neither question yields an easy answer.

As it happens, several pieces tackling this head-on appeared in quick succession. Linus Torvalds, in his [Open Source Summit keynote](https://thenewstack.io/torvalds-ai-programming-productivity/), said the claim "AI wrote 99% of my code" makes him angry, and shot back that 100% of those people's code was in fact written by a compiler. His position: AI is just the next step in the abstraction lineage of machine code → assembler → compiler, and only people who understand the system get good results from it. Jimmy Koppel argued in [Software Design in the Age of AI](https://self-service.mirdin.com/software-design-in-the-age-of-ai) that as low-level implementation gets automated, high-level design skill becomes the core human advantage. Josh Collinsworth came at it [from the opposite direction](https://joshcollinsworth.com/blog/productivity), asking whether AI actually delivers productivity or merely manufactures the _feeling_ of being productive, and warning how that very advantage gets eroded. He describes blasting through a backlog with agents, then looking back and realizing that while the codebase may have improved, he hadn't, and that he couldn't defend the PRs he himself had opened. And it's not just a feeling. In METR's [controlled study](https://arxiv.org/abs/2507.09089) of experienced open-source developers, tasks done with AI tools were actually 19% _slower_, while participants believed they had been 20% faster. With only 16 participants the sample is too small to generalize from, but it demonstrates one thing clearly: perception and reality can point in exactly opposite directions.

Compress the question running through all three pieces into one sentence and you get this.

**If the code meets spec and the bugs get fixed, does it matter that you can't read or explain it?**

More concretely: if QA passes it, the PM approves it, and the customer is happy, does it matter that the person who produced the code can't read it?

The intuitive answer is "well, you should still be able to read it." But intuition is not an argument.

## For: We Already Live Without Reading Code

Let's defend the "it doesn't matter" position properly. Setting up a weak version just to knock it down would be pointless. The strongest form of the argument goes like this.

**First, we already live this way.** In a frontend app, the code you wrote yourself is less than 1% of the total. The rest is node_modules, the browser, V8, the OS, compilers, CPU microcode. Nobody has read this enormous pile of code. We verify all of it purely through externally observable behavior. Does it work, does it pass tests, does a maintainer respond when something breaks. Nobody reads AWS's code; we trust it in exactly the "customer says OK" way. Software engineering has run on trust-by-interface, not trust-by-understanding, from the very beginning. The claim "you must be able to read the code" selectively applies to the last 1% a rule we already decline to apply to the other 99%.

**Second, reading code was never the goal; it was a proxy signal.** The real goals of engineering were always correct behavior, manageable cost, and ease of repair. Reading code was simply the cheapest tool for estimating behavior in an era when testing was expensive. Once directly checking behavior becomes cheap and dense enough (automated tests, type checking, canary deploys, observability, customer feedback loops), the proxy signal loses its usefulness. We ourselves enshrined encapsulation and information hiding as principles of good design. After decades of teaching that good design means not having to read the implementation, saying that we suddenly must read implementations because AI writes them is a betrayal of our own principles.

**Third, the history of engineering is a history of abandoning lower layers.** Machine code → assembly → C → garbage-collected languages → frameworks. At every transition there were people insisting "a real engineer must know the layer below," and every time, economics overruled them. Developers who don't know memory management became the mainstream, and nothing bad happened. You can even borrow Torvalds's own logic here. Nobody says "the compiler wrote 100% of my code," and nobody verifies compiler output by reading it. We trust the toolchain and the tests. Even repair stopped requiring understanding. When the cost of generation approaches zero, regenerating a broken thing beats surgically patching it. Add the failure case to the spec as a test, regenerate, re-verify. This logic has already won in infrastructure. Nobody fixes a broken container. They redeploy. [Cattle, not pets](http://cloudscaling.com/blog/cloud-computing/the-history-of-pets-vs-cattle/): do you hand-tend and repair servers like named pets, or treat them as a herd, discarded without sentiment and swapped for identical new instances the moment they break? [Immutable infrastructure](https://martinfowler.com/bliki/ImmutableServer.html) (never modify a running server, only replace it wholesale) was precisely one side of this debate, and that side won. The conclusion was that hand-tuned servers become "snowflakes" whose state nobody can reproduce, and are therefore more dangerous. AI is the next layer, and there's no reason code should be different.

**Fourth, the bugs that "reading would catch" aren't caught by reading either.** This answers the most common objection: that security vulnerabilities, race conditions, and bugs that surface much later slip past QA. Those bugs slip past code review too. The [research](https://dl.acm.org/doi/10.5555/2486788.2486882) on defect detection in human code review suggests that review's actual output is closer to readability and knowledge sharing than defect discovery; it catches shallow problems but misses concurrency and security bugs. What actually catches this class of bug was never reading; it was fuzzers, static analysis, property-based testing, canaries, observability, penetration testing. All tools that only reveal problems by running the thing. The choice isn't "reading vs. defenselessness"; it's "fallible human eyes vs. stronger verification machinery," and the resources you'd spend teaching everyone to read code are better spent strengthening that machinery.

**Fifth, understanding doesn't scale; verification and accountability do.** Code output is exploding while the amount a human can read stays fixed. If your safety strategy depends on "a human reads it," safety only degrades as code volume grows. Desirable or not, it's a losing strategy. Tests and monitors scale with compute. So does accountability: even in regulated industries, the explainability regulators demand is satisfied not by understanding in an engineer's head but by audit trails, decision logs, reproducible builds, and documented test evidence. Accountability is a matter of contracts, not of mental comprehension. Banks have been running COBOL nobody fully understands for decades, passing audits on process evidence. Betting on the side that can scale is the engineering choice.

That's the best case for the "for" side. Honestly, it's quite strong. The first and fifth points in particular are hard to refute head-on. But the argument as a whole has one structural flaw, and that flaw is decisive.

## But: The Conditional's Premise Is Circular

Go back to the original question. "If the code meets spec and the bugs get fixed." Pull this conditional apart and it collapses in five places.

### 1. Who knows the spec is satisfied, and how?

What a person who can't read code can know is "the tests pass," not "the spec is satisfied." These two diverge precisely where it's expensive. Tests only verify the cases their author thought of, and a large share of real-world bugs are not spec violations but **gaps in the spec**. Race conditions, boundary values, security vulnerabilities, data-consistency breakage that surfaces days later.

More fundamentally, QA and customers can only observe **what happened**; they cannot observe **what must never happen**. "PII never appears in the logs," "retries never produce duplicate charges": the things that must not occur are invisible to acceptance testing. A customer cannot "OK" the absence of an injection vulnerability. It isn't visible.

In other words, verifying that the premise "if the spec is satisfied" is even true itself requires understanding the system. The question smuggles its answer into its own premise.

### 2. Repair can be outsourced; detection cannot

"Can fix the bugs" quietly presupposes "noticed there was a bug." The fixing can be handed to AI. But the noticing that something is wrong (especially the sense that something is off even though every test passes) does not arise without understanding the system. The bottleneck is not repair but detection, and the conditional skips exactly that bottleneck.

The archetype of this kind of bug is what Koppel calls hidden coupling. Two pieces of code that look distant and unrelated are in fact entangled, hiding without a compile error or a test failure until some unrelated change sets them off. If you can't read code, you can't even know such a bug exists, let alone fix it.

### 3. The conditions that made the precedents work are absent here

Look again at the "for" side's strongest weapon, the precedent argument: compilers, node_modules, AWS. Every one of those precedents had at least one of two things.

- **The meaning of the input is preserved intact.** A compiler carries a decades-verified guarantee that it faithfully preserves the semantics of the input I understood (the source code). A wrong compiler makes the news.
- **A responsible counterparty who holds the understanding.** When an npm package breaks, a maintainer fixes it, or I fork it with my own understanding. When a SaaS breaks, the vendor is on the hook for an SLA.

AI-generated code is the first code humanity has depended on with **neither**. The input is natural language, which cannot fully carry intent, and there is no guarantee the intent is preserved. On the other side sits not a responsible person but a machine that answers differently each time. This is why Torvalds's compiler analogy, rhetorically powerful as it is, is only half right. A compiler produces the same output for the same input and its correctness is guaranteed; an LLM does neither. The "both are abstraction tools" frame papers over this difference in verification cost. The precedents do not apply.

That is why in regulated domains this debate was closed before it started. Even Anthropic's recently published [Zero Trust for AI Agents](https://claude.com/blog/zero-trust-for-ai-agents) guide states flatly that in industries handling financial, health, or personal data, the ability to trace and explain every agent action back to its triggering input is not optional. The "for" side's point that accountability is contractual is correct, but to _produce_ the audit trails and explanations that satisfy that contract, someone has to understand the system. "The tests passed, but nobody knows the code" is not an answer you can give a regulator.

### 4. Verification is a moment; ownership is a duration

The OK from QA, PM, and customer is a single photograph of one moment: a specific point in time, the tested paths, today's load. Owning software is a continuing obligation toward everything that arrives **after** that gate. The security incident, the 10x traffic spike, the cost of changing things for the next feature.

What the PM approved is the current feature, not the cost of modifying it for the next one. That cost is determined by code structure the PM cannot see. A codebase that passes every acceptance gate while simultaneously becoming unmodifiable is entirely possible. In a setup where nobody answers for what comes after the gate, "we got the OK, we're done" is not the completion of verification but a vacuum of responsibility.

### 5. The regeneration loop doesn't remove understanding: it relocates it

The "for" side's "regeneration replaces repair" logic circles back to the spec. To add the failure case to the spec and regenerate, you have to know what failed and what to pin into the spec. That is system understanding.

And Collinsworth described what empirically happens when you loop repair and regeneration without understanding. Unread patches pile up, codebase complexity climbs, and as complexity climbs the LLM's own effectiveness declines. Koppel made the same observation: when a contractor at his company moved to a firm with lower code quality, Claude Code's effectiveness dropped sharply. The METR result we saw earlier (slower, while feeling faster) is exactly what makes this loop dangerous. Even as effectiveness falls, you believe you're speeding up, so you never receive the signal to stop. There is no guarantee anywhere that the loop keeps spinning forever. If anything, the loop is structured to eat its own efficiency.

Compressed into one sentence:

**The ability required to verify that the conditional is true is precisely the ability the conditional claims is unnecessary.**

## Understanding Doesn't Disappear, It Changes Form: The Verification Layer

So does the "against" side win? It's not that simple. The "for" side's first, third, and fifth arguments (we already skip 99% of the reading, engineering history is a history of shedding layers, understanding doesn't scale) remain valid. The trend of reading less code line-by-line, day-by-day, is undeniable.

My conclusion is this. The "for" side is right about the direction; the "against" side is right about the timing and the form. Understanding is not eliminated; it moves from "reading code as a daily routine" to "knowing what must be verified." And someone who has the latter can still do the former when needed. It's the same muscle.

Sketch the destination of this move concretely, and in a pipeline where AI does the generating, the role becomes **designing and owning the entire apparatus that judges whether the output is right or wrong**. I've been calling this the verification layer.

Intent → Spec → **[Generation: AI]** → Static gates → Test gates → Selective human review → Canary → Runtime monitors → Incident response

Only the generation step belongs to AI; everything else is the verification layer. It decomposes into six concrete jobs.

**1. Writing specs so they are verifiable.** The key is not "what it should do" but "what must never happen." "The order API must be idempotent," "retries must never produce duplicate charges." AI builds what you ask for; it cannot guarantee it avoids what you didn't ask about. Knowing what must not happen is system understanding, and only when that is pinned into the spec does "tests pass ≈ correct" hold.

**2. Test strategy design.** Not writing test code; AI does that. The problem is that when AI writes tests to fit its own code, you get circular verification: grading your own exam with your own answer key. The verification layer's job is designing the break in that circle. Property-based tests for invariants that must always hold, contract tests at the boundaries, mutation testing for the quality of the tests themselves. It's deciding which verification tool to attach to which class of failure, and if that pairing is wrong, ten thousand tests still leave holes.

**3. Selective deployment of reading (triage).** Not reading everything: deciding where to read. Authentication, paths where money flows, concurrency, data migrations, dependency changes: place human eyes on the high-blast-radius 5% and leave the rest to automated gates. This risk classification itself requires system understanding. The paradox (you must know how to read code in order to judge which code you don't need to read) becomes, here, a job description.

**4. Runtime verification design.** Verification doesn't end at merge. Canary deploys, alerts that fire when behavioral invariants break, and the measurement of time-to-detection and the fraction of fired alerts that were actually investigated. It's weaving the net that catches, in production, the failures static verification can't.

**5. Verifying the verifiers.** The agents inside the AI workflow (review agents, test-generation agents) are themselves subjects of verification. Agreement rates with human reviewers, false-positive rates, and criteria for pulling the plug when they fail. The review agent I'm building has an eight-week evaluation deadline attached: if I can't demonstrate its effect in numbers, it comes down.

**6. The last line of defense.** When all five of the above are breached: the ability to go down to the code yourself, read it, and debug it. You don't use it day to day. But its existence is what makes everything else trustworthy. Without it, items 1 through 5 aren't verification; they're just process management. In the age of robotic surgery, the surgeon operates less often; but if someone who can't operate supervises the robot, that's not a doctor, that's an administrator, and the moment complications hit, the patient dies.

### And there's now one more system to understand

There is one way the verification layer differs from the pre-AI senior role: there are now two systems to understand. **The software system (the artifact), and the system that produces the software (the generator).**

Before AI, the thing writing the code was a person, and we knew intuitively how people fail. They slip up when tired, misread the ticket, phone in the tests when they can't be bothered. A reviewer's instincts stood on that failure model. Now the thing on the producing side is a machine that answers the same question differently every time, and its ways of failing are nothing like a person's. It fabricates with confidence; it builds what it's told but doesn't know the untold "must-nots"; it gravitates toward the most common, most ordinary design in its training data; it writes tests to fit its own code, grading its own answers with its own answer key; and it is structurally weak to hidden coupling.

The incident I mentioned earlier is the archetype. Logs were accumulating at dozens of times the normal rate, but only in one specific development environment, and I had an agent hunt for the cause. The agent burned through tokens working down the "usual suspects for duplicate events" (IntersectionObserver, fetch, and so on), each one a miss. The real cause was none of that. Next.js passes a phase value like `PHASE_DEVELOPMENT_SERVER` as the first argument to the config function, and a single botched phase comparison was sending that one environment down an unintended config branch. That was all. I found it only by reading the code myself.

What this incident shows is the model's mode of failure. The model wasn't tracing causality; it was cycling through patterns that co-occur with "duplicate logs" in its training data. It produces a long, plausible list of suspects, but it cannot solve the case where the culprit is one boring conditional. And through it all, the agent _looked_ remarkably productive: it was always investigating something. The gap Collinsworth described between perception and reality takes exactly this shape in debugging.

There's one more case like it. I handed an agent an entire Next.js codebase and asked it to improve performance, with no further guidance. The agent diligently combed through server and client code and came back with micro-optimizations: added memos, dynamic imports, unnecessary lazy initialization removed. None of it was wrong, but none of it was the point, and the agent itself concluded these changes would have negligible impact, ultimately declaring the project "already well designed for performance." The real problem wasn't in the code at all; it was in the dependency tree. Duplicated packages and mismatched versions were shipping multiple copies of the same libraries into the bundle, and that was the main driver of bundle size. [As I covered in an earlier post](https://yceffort.kr/2026/05/pr-diff-vs-bundle), bundle cost is invisible at the individual-file level. The agent reads code file by file, but how the bundle assembles is a property visible only from the whole dependency graph; it isn't written down in any single file. The verdict "well designed" was true only within the layer the model could see, and the model, unaware that layers existed beyond its sight, reported its field of view as the whole.

The interesting part: if I had handed the same agent the lockfile and said "find the duplicate dependencies," it probably would have found them easily. The problem wasn't capability; it was aim. Faced with a vague goal like "improve performance," the model hunts for optimizations in the unit it can see (individual code) and cannot judge which layer the problem lives in. That judgment is system understanding, and without a human setting the aim, the model's capability gets spent in the wrong place.

The two cases fail in different directions. The first is a failure of depth: it couldn't trace causality to the end and cycled through plausible patterns instead. The second is a failure of breadth: it couldn't locate the layer the problem lived in and dug only in the units it could see. They share one thing: both could only be corrected by a human who understood the system.

There's a predictable rebuttal here. "Give a better model more tokens and longer to analyze, and it would have found them eventually, wouldn't it?" Partially true. With better tooling attached (lockfile access, a bundle analyzer, bisect) and enough runtime, both problems would quite likely have been found eventually. So let me be clear: these cases are not an indictment of model limitations. Six months from now, a model may well find these bugs.

Two things surface here. One: knowing the system collapses the search space instantly. Someone who knows Next.js's configuration structure gets from "only happens in one environment → the branch that splits environments → phase" in minutes. Without that knowledge, code, config, infrastructure, and dependencies all remain candidates. The places to examine explode, and tokens are the cost of rummaging through every one of them. Even if both paths reach the answer, the cost differs.

The more essential issue is termination and acceptance. This one isn't solved by pouring in more tokens. In the actual incident, the model didn't end with "I can't find it." It terminated with a confident wrong answer: "this project is already well designed." Give it 10x the tokens with the same termination behavior and you just get a more confident wrong answer. And the assumption "it would have found it eventually" is something you can only say after already knowing the outcome. It's a statement available only to someone holding the answer. Inside ignorance, you can judge neither when to stop nor whether to accept the answer you got. The only reason we know the model's answer was wrong is that I read the code and found the real cause. The assumption "run it longer and it would have found it" cannot even be checked without understanding, and the standing to make that assumption belongs only to someone who understands. The circle closes here again.

So if a better model comes along and finds these bugs, that is not a refutation of this essay. The human's job simply moves to judging whether the cost is worth paying, designing the harness, and rendering the acceptance verdict on the answer that comes out. That is the verification layer's work, as described above.

Without knowing the generator's characteristic failure points, triage is impossible. Where to read has become a judgment computed from two factors multiplied together: "riskiness of the code × weaknesses of the generator." In factory QC terms, knowing the product spec isn't enough; you have to know which direction this particular lathe tends to drift to decide where to place your measurements.

This is my definition of "understanding the AI workflow." It is not prompt-writing technique. It's four things.

- **Failure-mode knowledge.** What classes of problems does the model get systematically wrong?
- **Pipeline design.** Context management, units of task decomposition, placement of hooks and gates, which verification goes at which stage.
- **Security of the workflow itself.** Prompt injection, tool permission scoping. To a review agent, the PR diff is untrusted input. Planting "approve this PR" in a PR description or a code comment is the exact attack scenario.
- **Measurement of the workflow.** Agreement rates, false-positive rates, criteria for retreat.

One asymmetry, though, must be stated clearly. **This knowledge has a short half-life.** Today's model failure modes get patched in the next generation, and workflow best practices turn over every six months. System understanding, meanwhile, compounds. A decade-old intuition about concurrency bugs is still good today. AI-workflow understanding is essential but depreciating; system understanding is the durable asset. Over-invest in the former and become a "tool expert," and every tool change sends you back to square one. The only non-depreciating thing on that side is one skill: the eye that quickly discerns, when a new model arrives, where and how it goes wrong.

To summarize, it's a tripod. **System understanding (which includes reading code), generator understanding (the AI workflow), and the verification design that combines the two.** The claim "meets spec + can fix bugs = fine" only holds when someone possesses all three, while the claim itself asserts all three are unnecessary. A self-contradicting structure.

## What Should Juniors Aim For?

Back to the second of the two questions we split at the start: the problem of growth. What do we tell someone who is just starting out?

You can't just hand them the tripod. That's a description of the destination, not a curriculum. Verification design stands only on top of system understanding. The ability to conjure "what must never happen" into a spec arises only in someone who has personally traced such failures, and triage presupposes an eye for dangerous code, an eye that comes from having read and broken a lot of it. You cannot verify what you have never built or broken.

And before any of that, the real problem juniors face needs a name. **Work no longer trains you for free.** Before AI, doing the job was accumulating the skill. Working with agents now, the better you perform the job, the less you may be learning. Collinsworth's question ("if juniors learn nothing for three years, where do seniors come from?") is exactly this problem. Work used to be the training. Now the training has to be designed separately.

If I had to give the goal in one sentence, it's this.

**Don't be the person who writes faster than AI. Be the person who notices fastest when AI is wrong.**

Generation speed was handed out equally to everyone, so it's no differentiator; the ability to detect wrongness doesn't emerge without training, so it is. Broken down into rules of behavior, there's an order to it.

**1. Never merge code you can't explain.** If only one rule survives, it's this one. For every PR, you must be able to explain to a reviewer "why it's done this way, and what could go wrong." The beauty of this rule is that its enforcement is natural. It doesn't ban AI use, and it forces reading at exactly the points where reading is needed. It structurally forecloses the state of "I opened the PR but couldn't defend it."

**2. Volunteer for debugging.** Incidents and root-cause analysis are the last natural training ground left in the AI era. Debugging forces reading, forces hypothesis formation, forces system understanding. And it can't be outsourced yet. When an incident hits, don't spectate; go in, and read every postmortem. The ability to know what can go wrong comes not from lectures but from having personally chased down things that went wrong.

**3. Own something, however small, through maintenance.** Build-and-abandon side projects now carry almost no training value, because generation is free. The value is in the maintenance phase: running something for six-plus months and taking the hits from your own bugs. Concepts like hidden coupling and spec gaps can't be learned from writing about them; they're internalized when your own code comes back as a boomerang. Operating one thing for a long time is a better goal than building ten things fast.

**4. Keep a log of cases where AI was wrong and you caught it.** Generator understanding requires almost no separate study. Use agents daily and maintain one log: "what the model was confidently wrong about this week, and how I noticed." This is the training in doubting and verifying AI that a junior can start today, and it interlocks naturally with rule 1, because catching wrongness requires reading.

There's one trap to warn about. Juniors will want to skip rule 1 and go straight to workflow mastery, because that path looks modern and marketable. Worse, in the short term the market rewards it. Blasting through tickets with agents looks productive. But workflow mastery without system understanding is process management without a last line of defense, and the first major incident exposes it for what it is. By then, years have passed.

Finally, this is not homework for juniors alone. Work used to enforce this ordering automatically, and now it doesn't, so **enforcing the ordering has become a design responsibility of seniors.** Actually asking "why did you do it this way?" in review. Deliberately putting juniors on incident response. Making merge explanations a cultural norm. Half of giving juniors a goal is seniors building the environment where that goal can function.

## Closing

Let's answer the opening question. If the code meets spec and the bugs get fixed, does it matter that you can't read or explain it?

Honest concession first: a domain where it genuinely doesn't matter exists. Software that is short-lived, isolated in blast radius, whose failures are immediately visible and cheap. One-off scripts, prototypes, internal tools. Insisting "you should still understand it" in this domain is romanticism, and this domain is not small; it's growing.

But outside that boundary (software that lives long, has money riding on it, and fails in ways that surface much later) the conditional itself doesn't hold. Confirming the spec is met, discovering the bugs, and judging which code this logic may safely be applied to all require understanding. Even the judgment "this is code that doesn't need to be understood" (the sorting itself) requires the ability to understand. Someone who can't read code can't even scope their own question.

So the final answer is this.

**Code that needs no understanding exists. But only someone capable of understanding can draw the line around it.**

The frequency of reading may fall. The ability to read must not.

> What happens if we take this question beyond code? The next essay applies the same logic to the boundaries between product, engineering, and design as job roles: [If AI Erases the Lines Between PM, Engineer, and Designer, What Remains?](https://yceffort.kr/2026/06/when-job-titles-blur)

## References

- [Linus Torvalds Talks AI, Vibe Coding and Programmer Productivity](https://thenewstack.io/torvalds-ai-programming-productivity/) - The New Stack
- [Software Design in the Age of AI](https://self-service.mirdin.com/software-design-in-the-age-of-ai) - Jimmy Koppel
- [Is AI Actually Making Developers More Productive?](https://joshcollinsworth.com/blog/productivity) - Josh Collinsworth
- [Measuring the Impact of Early-2025 AI on Experienced Open-Source Developer Productivity](https://arxiv.org/abs/2507.09089) - METR
- [Expectations, Outcomes, and Challenges of Modern Code Review](https://dl.acm.org/doi/10.5555/2486788.2486882) - Bacchelli & Bird (ICSE 2013)
- [Zero Trust for AI Agents](https://claude.com/blog/zero-trust-for-ai-agents) - Anthropic
