---
title: 'If AI Erases the Boundaries Between Planning, Development, and Design, <em>What Remains</em>?'
tags:
  - ai
  - essay
  - software-engineering
  - career
published: true
date: 2026-06-12
updated: 2026-10-19 13:52:52
description: 'The claim that job boundaries are collapsing is only half true. What collapses is production; judgment and responsibility remain.'
series: 'Judgment in the AI Era'
seriesOrder: 2
art:
  undraw: hr-presentation
  layout: bauhaus
  hue: slate
  tone: light
  hero: 'T자형 인간'
---

## Overview

"The barrier to programming has come down incredibly low. Everybody is now a programmer. You just have to talk to the computer." That's what Jensen Huang said in his [2023 Computex keynote](https://fortune.com/2023/05/30/nvidia-ceo-jensen-huang-everyone-programmer-with-ai-chipmaker-taipei-computex/). In an [interview with Alexis Ohanian](https://fortune.com/2024/02/04/sam-altman-one-person-unicorn-silicon-valley-founder-myth/), Sam Altman said there's an ongoing bet in a group chat with his tech CEO friends over when the first one-person, billion-dollar company will emerge. Something once unimaginable without AI has now become something that's about to happen.

This doesn't sound like mere stage hyperbole, because the actual landscape of the workplace is moving in that direction. A PM has an agent build a working prototype in a single day. A designer writes their own components and opens a PR. A developer produces mockups and writes copy. The boundary that, until just a few days ago, was drawn with "that's not your job" is quietly blurring. The statistics point the same way. In Figma's [2025 report](https://www.figma.com/blog/2025-shifting-roles-report/) surveying 1,199 people who build products, 64% said their job spans two or more disciplines, and more than a third said three or more. The number of distinct tasks people perform in a year grew 17.5% year over year.

The discourse has already gone a step further. Claire Vo, creator of ChatPRD, wrote in [Product Management Is Dead](https://www.chatprd.ai/blog/product-management-is-dead) that the traditional triangle of planning, design, and development is being replaced by "generalist teams that cover multiple domains." Dan Shipper argued in [The Knowledge Economy Is Over. Welcome to the Allocation Economy](https://every.to/chain-of-thought/the-knowledge-economy-is-over-welcome-to-the-allocation-economy) that the essential nature of work is shifting from "the person who builds directly" to the "model manager" who delegates work to AI and harvests the results. In this coming era, you're evaluated not by how much you know, but by how well you delegate.

So the same conclusion echoes everywhere. Job boundaries are outdated. They should be erased. Everyone becomes a builder.

This statement is only half true. And the half that's false matters more.

In my [previous post](https://yceffort.kr/2026/06/do-you-need-to-read-code), I wrote that reading code isn't disappearing; it's shifting into "knowing what needs to be verified." The same structure applies to job boundaries. There is one thing that collapses and one thing that remains, and if we fail to distinguish between them, the fine slogan "let's erase the boundaries" turns into a dangerous organizational design. Boiled down to a single question:

**If anyone can now build anything, is it okay to erase the boundaries between planning, development, and design?**

Let's first build the strongest possible case for "yes, erase them." There's no point knocking down a weak version of the argument.

## In favor: the boundary was a partition built by tools

**First, much of the boundary between job functions was a byproduct of tool proficiency.** You had to know how to write code to be a developer, know how to use Figma to be a designer, know how to write spec docs to be a PM. That "do you know how to build it" was the barrier to entry, and the barrier was the boundary itself. But this barrier wasn't essential; it was a limitation of the tools. Just as drawing required hands, building a screen simply required code. When AI stands in for those hands, the barrier disappears, and the boundary built on top of it collapses along with it. This is exactly what Jensen Huang meant. In an era where you just have to talk to a computer, the line that once separated people by "can you do it" loses its meaning. And the Figma numbers we saw above (64% working across two or more disciplines) prove this isn't a prediction. It's already happening.

**Second, the real cost of boundaries was the handoff.** Half the reason products move slowly is the translation and waiting between disciplines. Planning hands off a document, a designer draws mockups, development receives and implements them, and then it's passed to QA again. At every stage, intent leaks out, and everyone waits in line. Every time a handoff occurs (the transfer of deliverables between disciplines), the intent that lived in someone's head gets lossy-compressed: into a document, then into a mockup, then into code. If one person carries the intent all the way through, this loss disappears entirely. AI makes that possible. Erasing the boundary isn't about doing sloppy work; it's an engineering choice to drive translation cost to zero.

**Third, integration isn't a new phenomenon; it's a recurring pattern.** Frontend and backend were once separate disciplines, but around 2010 the term "full stack" spread and eventually became mainstream. Development (Dev) and Operations (Ops) were once split by departmental walls, but Amazon's Werner Vogels already declared those walls torn down in a [2006 ACM Queue interview](https://queue.acm.org/detail.cfm?id=1142065): "The traditional model is that you take your software to the wall that separates development and operations, and throw it over and forget about it. Not at Amazon. You build it, you run it." Twenty years later, nobody disputes that DevOps won. Every time, there were people saying "those are different disciplines," and every time, the side that integrated built faster and better products. The collapse of job boundaries isn't an aberration; it's a normal convergence that repeats whenever tools get good enough. AI is just pushing that convergence one step further.

**Fourth, small teams beat boundary-bound organizations.** Linear builds a single feature with one designer and two or three engineers. [There's no dedicated PM](https://www.lennysnewsletter.com/p/how-linear-builds-product), and planning work is distributed across engineers and designers. The product built this way is widely regarded as among the most polished in the industry. Armed with AI, this shrinkage doesn't stop here; three or four people, or even one person, will build what used to take twenty. Altman's one-person-unicorn bet is the endpoint of this trend. Boundaries were originally a device that growing organizations created to manage collaboration, but AI eliminates the need for that scale in the first place. The boundary isn't disappearing; the reason it was needed is.

**Fifth, even judgment is increasingly becoming a tool.** Let's preempt the most common objection: "AI does the producing, but judgment is still a human job." That line of defense is being pushed back too. Code review already gets a first pass from agents, accessibility violations get caught automatically, design mockups get AI critiques attached, and gaps in a PRD get flagged by AI. If judgment is the core of each discipline, then even that judgment is being turned into a tool. This is exactly the stage Shipper's allocation economy is describing. Human work retreats one more step, from making the judgment to delegating the judgment. So the comfort of "judgment remains" gets thinner with each passing year.

That's the strongest case in favor. It's quite compelling. Points two and five, in particular, are hard to rebut head-on. But the entire argument is stuffing two different things into a single word: "boundary."

## However: the word "boundary" is conflating two different things

There's the boundary of **who can build**, and there's the boundary of **who judges and takes responsibility for what's correct**. These two don't move together. The first collapses. The second remains. Let's revisit each pro-side argument.

### 1. The ability to build and the ability to know what's correct are different abilities

Even if AI produces the code, knowing whether that code is correct (whether it's safe, whether it can be maintained later, whether things that shouldn't happen won't happen) remains a human task. The same distinction that held for code applies equally to job functions.

The real core of each discipline was never the manual skill of producing a deliverable, but the judgment to determine whether that deliverable is correct. The core of design isn't the hand that draws a pretty mockup, but the eye that knows whether this screen actually works in front of real users: whether it breaks under edge states, whether accessibility is preserved. The core of planning isn't writing documents, but the judgment of what needs to be built and what must never happen. The core of development isn't typing code, but the instinct for knowing where the system will break.

The first pro-side argument ("the boundary was a partition built by tools") is correct for the boundary of manual skill, but wrong for the boundary of judgment. Actual usage data supports this distinction too. Anthropic's [Economic Index](https://www.anthropic.com/news/the-anthropic-economic-index), which anonymized and analyzed millions of Claude conversations by occupation, found that 57% of AI usage was augmentation (assisting a person's work) while 43% was automation, delegating the task entirely. And occupations that delegated more than 75% of their work to AI made up only about 4%. This isn't a picture of entire disciplines being replaced wholesale, but one where production work within a discipline gets eaten away first, while judgment work remains.

What's interesting is the labor market's response. If production is being automated, you'd expect demand for designers to fall. But in Figma's [2026 report](https://www.figma.com/reports/design-hiring-study-2026/) surveying hiring managers, 47% said demand for designers had increased (82% said it had increased or stayed the same). According to an estimate from Designer Fund, a design-focused VC cited in the report, design job postings at portfolio companies rose roughly 60% year over year in 2025. We should account for the fact that this is a survey run by a company that sells design tools. Still, what's worth noting is the _content_ of that demand. More than 45% of hiring managers said they're looking for collaboration, systems thinking, and product strategy (all judgment-side capabilities) rather than visual polish. As production got cheaper, the market started pricing judgment higher.

### 2. The handoff was both a cost and a verification

The second argument is the most persuasive one, and that's exactly where the trap lies. The boundary between disciplines was friction, but it was also a **mutual checkpoint**. As a designer receives intent handed off by planning, they filter out "users won't understand this." As development receives a designer's mockup, they filter out "this state will break on implementation." The handoff wasn't just a waiting line; it was a gate where different kinds of judgment checked each other.

When one person carries intent all the way through, the translation cost disappears. But so does that verification. A structure where you write your own plan, design it yourself, build it yourself, and approve it yourself is exactly like the loop I described in my previous post: AI writing tests to match its own code, grading its own answer with its own answer. Where friction disappears, what remains isn't just speed; a blind spot also remains, one that nobody on the other side is checking.

Actual collaboration data points not toward boundaries vanishing, but toward them getting denser. In Figma's [survey of designers and developers](https://www.figma.com/reports/designer-developer-trends/), the share of developers who collaborate daily with designers doubled from 16% in 2023 to 32% in the 2025 report. The more disciplines mix, the more often people end up borrowing each other's judgment. Erasing the boundary and crossing it more often are two different things.

### 3. Precedents never removed responsibility

The history behind the third and fourth arguments is real. But look closely at that history, and it points in the opposite direction from what the pro side claims. Vogels' "You build it, you run it" erased the _boundary_ between development and operations, not the _responsibility_. In fact, quite the opposite. In the same interview, Vogels said: "Giving developers operational responsibilities has greatly enhanced the quality of the services." The core of the integration was that the builder _took on_ operations as well. The wall wasn't torn down; both sides of it were placed on one person's shoulders.

Full stack is the same. Someone who does both frontend and backend carries the responsibility of both domains. Linear's "no PM" holds up under scrutiny too. There's no PM _title_, but the PM _judgment_ hasn't evaporated: every project has a lead, and that lead owns the judgment about what to build. Planning work wasn't distributed away; it was reassigned to people capable of making that judgment.

So what these precedents actually prove isn't "it's fine to erase boundaries"; it's "it's fine to erase boundaries only when responsibility can be concentrated in one person." When "let's erase the boundary" degrades into "let's erase the person responsible for each judgment," the opposite happens: everyone can build everything, but nobody makes the final call on what's correct. Every gate gets passed through, but there's no one standing behind the gate to answer for it: a vacuum of responsibility opens up at the organizational level.

### 4. Tooling judgment doesn't replace owning judgment

Here's the answer to the fifth argument. AI _assisting_ judgment and AI _owning_ judgment are different things. AI can list out accessibility violations. But deciding which items on that list actually matter for this product, what to sacrifice now and what must be blocked, and whether the AI's critique itself is even correct, that's still a human task. Just as building a verifier creates the need to verify the verifier, turning judgment into a tool creates the need to judge whether to accept that tool's judgment. It can retreat one step back, but it doesn't disappear.

The allocation economy the pro side cites actually concedes this point. Shipper's conclusion isn't "judgment becomes unnecessary"; it's "you're evaluated by how well you delegate." What to delegate and what to hold onto yourself, whether the delegated result is any good; the word "allocation" is itself just another name for judgment. Becoming a model manager isn't being freed from judgment; it's stripping everything away except judgment.

### 5. Mistaking the ability to produce for the ability to know

So the real danger, prior to any organizational structure, is an illusion an individual falls into: mistaking the fact that you can now generate a designer's deliverable for having acquired a designer's judgment.

A PM who used an agent to build a screen hasn't gained design sense. He doesn't know that an edge state he's never seen is broken, doesn't realize accessibility is broken either, and declares "done" anyway. The state Collinsworth confessed to ("I couldn't defend the PR I'd submitted") replays across every deliverable, crossing every discipline. Generation came for free. Discernment didn't come with it.

Boiled down to one sentence:

**The ability to cross a boundary (production) and the ability to know which boundaries shouldn't be crossed (judgment) are different abilities.**

## Boundaries don't vanish, they get redrawn: the division of judgment

So does the "against" side win? It's not that simple. The direction favors the pro side; the shape favors the con side. The boundaries drawn along tool lines (the person who writes code, the person who draws mockups, the person who writes documents) really do collapse. In their place, boundaries drawn along judgment lines get redrawn: the person responsible for security judgment, the person responsible for UX judgment, the person responsible for judging what to build.

Why judgment-based boundaries? Because problems don't follow a discipline's home address. Just as a performance problem lives in the dependency graph rather than a code file, the cause of churn sometimes lives in a single line of copy rather than a feature, and the root of a design problem sometimes lives in the data model rather than the mockup. Deliverables come out organized by discipline, but problems ignore those boundaries entirely. Finding which box a problem's address belongs to: that's what depth does. And that's precisely the address-finding work that organizations, where everyone can now produce, need to divide up. It breaks down into roughly five things.

**1. Codifying judgment.** Turning your own box's judgment into a form the next box's person can use. Design principles, an accessibility checklist, a list like "duplicate charges must never occur on retry." You can't give everyone an expert's eye, but you can give them the checklist an expert looks at.

**2. Invocation protocols.** Deciding in advance how far someone can go alone, and where an expert must be called in. Call one in if the payment flow changes; call one in if you add a personal-data field, and so on.

**3. Selective expert review.** You can't put an expert on every single deliverable. So you place expert eyes on the small, high-risk set (the path where money flows, the first-time user experience, the feature that deletes data) and leave the rest to codified standards and automated checks. Classifying what's risky is itself a judgment belonging to that discipline.

**4. Redesigning the gate.** The more a team eliminates handoffs, the more deliberately it must decide where to place the second set of eyes that breaks the loop of grading your own answer with your own answer. The gate that used to arise naturally has disappeared, so now it has to be built. The Vercel example below is one form of this: no handoff, but two kinds of judgment meeting on the same deliverable.

**5. Boundary disciplines.** And the boundary itself becomes a job. Someone who holds deep judgment in both boxes: the market has already started naming this.

### The market is already redrawing the lines

Design engineer is that name. In a [2024 post](https://vercel.com/blog/design-engineering-at-vercel), Vercel defined this role as "someone with both an aesthetic sense and technical ability, who deeply understands a problem and then designs, builds, and ships alone." What's worth noting is the next sentence: "Instead of handing off a finished design, a designer sketches a starting point, and a design engineer refines it into a final design together, on Figma or in code." The distinction between disciplines hasn't disappeared. In the place where handoff used to be, a new discipline that holds both kinds of judgment in one body has emerged. The boundary wasn't erased; it was redrawn as a more expensive boundary. Because this role demands _both_ design judgment and engineering judgment.

There's an example from the opposite end too. In March 2025, a self-described non-developer founder who built EnrichLead, a SaaS product that cleans up sales lead data, bragged on X that he'd "built my entire SaaS with Cursor, zero lines of code written by hand," saying "AI isn't an assistant anymore, it's a builder." [Two days later, his tweet](https://pivot-to-ai.com/2025/03/18/guys-im-under-attack-ai-vibe-coding-in-the-wild/) began: "Guys, I'm under attack." API key usage was spiking toward the limit, people were bypassing payments, and the database was getting flooded with arbitrary data. The API key was hardcoded into the client, and the payment wall could be bypassed on the client side, things anyone with security judgment would have caught before launch. He eventually took the service down, writing: "You guys were right. I shouldn't have pushed insecure code to production."

What's worth dwelling on in this incident isn't the ending, which became fodder for ridicule, but the middle. He wasn't lacking in warnings. His original brag post got a string of security warnings in the comments, and he brushed them off. Without judgment, even warnings are just noise. Without an eye to tell a serious flag from mere nitpicking, information can arrive and still not be received.

There's a rebuttal one might expect here: "As models get better, won't AI handle security too?" That's partially true. A model six months from now will warn more forcefully against hardcoding API keys and will catch payment-wall holes more reliably. But this rebuttal conflates the _production_ of warnings with the _acceptance_ of warnings. What EnrichLead's founder lacked wasn't warnings; people had already given him plenty. What he lacked was the ability to judge whether to accept, dismiss, or how seriously to take each warning. As AI-generated warnings multiply, the burden of this judgment only grows heavier. Follow every single one, and nothing ships. Ignore every single one, and you become EnrichLead. The very spot my previous post called the problem of termination and acceptance reopens here, in the domain of job functions.

The market's answer is the same from both directions. Someone with deep judgment across both boxes gets a new job title and a higher price tag; production without judgment gets breached within days. The boundary isn't disappearing; it's being redrawn, shifting from the level of tools to the level of judgment.

One more thing worth adding: boundaries that disappear inside an organization reappear in the market. Even a one-person company with no job titles can't do all its judgment alone. It buys legal review, outsources accounting, commissions security audits. It's simply purchasing in the market the judgment it used to keep in-house. Even if Altman's one-person unicorn really does emerge, that company won't be one that does all its judgment alone; it will be one that knows which judgment to buy. And knowing what to buy is, itself, a judgment.

## What Should We Aim For

So which side should working professionals stand on? I don't think a flat generalist who does everything shallowly is the answer. Judgment can't be borrowed the way technique can. Design judgment only forms in someone who has personally watched a mockup fall apart in front of a real user, and the instinct for where a system will break only forms in someone who has personally traced down failures. The realistic destination is T-shaped: one deep column (a domain where you can take responsibility all the way to judgment) as the vertical stroke, with AI letting you cut shallowly across neighboring columns on top of it.

The origin of this idea is older than you'd think. A [1991 computing article in The Independent](https://wow.agiledata.io/wp-content/uploads/2022/10/David-Guest-1991-The-hunt-is-on-for-the-Renaissance-Man-of-computing.pdf) searching for "the Renaissance Man of computing" described 'T-shaped People' who combined information systems with management. The shape the computing industry was looking for 35 years ago has, in an era where production is free, become even more precise.

The vertical stroke has one property the horizontal stroke lacks: it survives when the tools change. The horizontal stroke is attached to the tool. How to use an agent, prompting tricks, this quarter's workflow: when the tool changes, these reset along with it. The vertical stroke isn't sitting on top of a tool. The eye that knows where users get lost, the instinct that knows where a system will break; these transfer intact even as the tools get overturned, from Figma to code, from code to agents. This is the difference between an asset that time erodes and an asset that time compounds, and it's one more reason to invest in depth.

The problem is that the T-shape doesn't grow on its own. The horizontal stroke is easy. AI widens it every day, and the reward is immediate. The vertical stroke is slow. It only grows in proportion to the time you've spent actually grappling with something. So left alone, everyone ends up as a flat line without a vertical stroke: shallow and wide only. A team that churns out output while nobody knows what's actually correct. If I had to give the goal in one sentence, it would be this:

**Move shallowly across neighboring columns with AI, but dig your own column deep enough to take responsibility all the way to judgment.**

Broken down into behavioral rules, it looks like this:

**1. Don't ship output under your own name that you can't explain.** This is the same rule I gave junior developers in my previous post, just extended from code to every kind of output. To ship a screen, you need to be able to explain "why this flow, and which users might trip up where." If you can't explain it, it's not your output; it's AI's output with your name lent to it.

**2. Cross into neighboring columns with a criterion for when to call in help.** Reaching into a neighboring column isn't a bad thing in itself, and you couldn't stop it even if you wanted to. The danger isn't in crossing over; it's crossing over without knowing where the edge of your own depth is. Make your own list of "from here on, I call in an expert," and then cross. Calling for help isn't something to be ashamed of; having that list at all is itself a skill. What EnrichLead's founder lacked was precisely this list.

**3. Dig your deep column where judgment accumulates.** Producing more output is no longer depth. In an era where production is free, depth only grows where judgment accumulates: reviews, incident response, user testing, operations. The job itself handles horizontal-stroke training for you; what you have to deliberately design for is vertical-stroke training.

**4. If you've erased the boundaries, redesign the verification.** This one falls on the organization. If a team lets one person carry a task from start to finish, it needs to build into its process a point where another discipline's eyes come in before shipping, and make the calling-in protocol a team agreement rather than an individual's conscience. Speed is easy to measure, and vanished verification isn't measured at all, so left alone, an organization will always drift toward erasing verification.

For someone just starting out, one more question remains: where should you dig your first column? In an era where discipline boundaries are shifting, asking which discipline to start in sounds like a reasonable question, but the answer isn't in the discipline. It's in wherever judgment accumulates fastest: that is, wherever failure is visible often and quickly. A small product where you're responsible all the way through operations, a team with rigorous reviews, a role where user reactions land on you directly. What matters isn't which discipline, but how short the feedback loop is. The vertical stroke ultimately grows only in proportion to how many times your judgment has been proven wrong.

One last trap worth warning about: in the short term, the market rewards the flat line. Churning out prototypes with an agent looks productive; crossing between disciplines looks modern. But width without judgment reveals its stagnation at the first major incident, and by then, years have already passed. EnrichLead's founder, too, _looked_ more productive than anyone, right up until the attack.

## Closing

Let's answer the opening question. Now that anyone can build anything, is it okay to erase the boundaries between planning, development, and design?

To concede honestly first: there is a real domain where erasing them is fine. Small, early-stage work where the cost of failure stays contained within itself. A first experiment in a solo venture, a prototype, an internal tool. Insisting on discipline boundaries here is stale bureaucracy, and this domain isn't narrow; it's growing. Altman's one-person unicorn is, in the end, a bet on just how far this domain can expand.

But there is a boundary. The moment payments get involved, the moment you handle someone else's data, the moment failure reaches an actual user, the work steps outside that domain. This is exactly where EnrichLead's founder went wrong, not in the code. He judged that he was still inside the safe zone, and the moment he started taking payments, he was already outside it. And what draws that line between inside and outside is, again, judgment.

Out in that world beyond, the answer splits. The boundary over who _can_ produce output collapses. It should collapse. The line that once divided people by whether they knew how to build something deserves to disappear. But the boundary over who judges and takes responsibility for what's correct remains. The cheaper production gets, the more expensive this becomes. My previous post ended with: "Code that requires no understanding does exist, but only someone who knows how to understand can draw the line marking where it applies." The same structure of sentence holds for disciplines too.

**Only someone with depth can draw the line between which boundaries should collapse and which should be defended.**

You're allowed to cross shallowly into neighboring columns. Just make sure your own column runs deep.

## References

- [Do you need to read code even if you don't need to understand it, as long as it satisfies the spec and fixes bugs?](https://yceffort.kr/2026/06/do-you-need-to-read-code) - Part 1 of this post
- [Nvidia CEO: 'Everyone is a programmer' with A.I.](https://fortune.com/2023/05/30/nvidia-ceo-jensen-huang-everyone-programmer-with-ai-chipmaker-taipei-computex/) - Fortune
- [Sam Altman wants AI to create a one-person unicorn](https://fortune.com/2024/02/04/sam-altman-one-person-unicorn-silicon-valley-founder-myth/) - Fortune
- [The Knowledge Economy Is Over. Welcome to the Allocation Economy](https://every.to/chain-of-thought/the-knowledge-economy-is-over-welcome-to-the-allocation-economy) - Dan Shipper
- [Product Management Is Dead](https://www.chatprd.ai/blog/product-management-is-dead) - Claire Vo
- [A Conversation with Werner Vogels](https://queue.acm.org/detail.cfm?id=1142065) - ACM Queue (2006)
- [How Linear builds product](https://www.lennysnewsletter.com/p/how-linear-builds-product) - Lenny's Newsletter
- [Design Engineering at Vercel](https://vercel.com/blog/design-engineering-at-vercel) - Vercel
- [Are Roles and Responsibilities a Thing of the Past?](https://www.figma.com/blog/2025-shifting-roles-report/) - Figma (2025)
- [Why Demand for Designers Is on the Rise](https://www.figma.com/reports/design-hiring-study-2026/) - Figma (2026)
- [Designer and Developer Trends Report](https://www.figma.com/reports/designer-developer-trends/) - Figma (2025)
- [The Anthropic Economic Index](https://www.anthropic.com/news/the-anthropic-economic-index) - Anthropic
- ['Guys, I'm under attack': AI vibe coding in the wild](https://pivot-to-ai.com/2025/03/18/guys-im-under-attack-ai-vibe-coding-in-the-wild/) - Pivot to AI
- [The hunt is on for the Renaissance Man of computing](https://wow.agiledata.io/wp-content/uploads/2022/10/David-Guest-1991-The-hunt-is-on-for-the-Renaissance-Man-of-computing.pdf) - David Guest, The Independent (1991)
