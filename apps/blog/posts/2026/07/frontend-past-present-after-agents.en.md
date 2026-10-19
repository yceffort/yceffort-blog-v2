---
title: 'Where <em>Frontend</em> Came From, and Where It Goes After Agents'
tags:
  - essay
  - ai
  - design-patterns
  - career
published: true
date: 2026-07-22 10:00:00
updated: 2026-10-19 13:52:52
description: 'Why the layers piled up, why we returned to the server, and why the stack survives even after agents. And why the survival of a stack and the value of the people who know it are two separate things'
art:
  undraw: launch-day
  layout: bauhaus
  hue: slate
  tone: light
---

## It Looks Like We're Back Where We Started

If I had to summarize the most advanced frontend practice of 2026 in one line, it would be this: render HTML on the server, ship only as much JavaScript as you truly need. Server Components, Astro, islands architecture. The names are new, but the shape isn't so different from 2008. After eighteen years, we've circled back to roughly the same spot.

Look closer, though, and it isn't the same spot. At the exact moment the pendulum returned, a more fundamental variable started to change: the hand that writes the code. Just as the twenty-year argument over client versus server was winding down, the very premise that a human writes the code began to wobble. So in this essay I want to work through three questions in order. Why did all those layers pile up (the past)? Where are we standing now (the present)? And what happens to this stack when agents write most of the code (the future)?

The body runs long, so here is the conclusion up front. Most of the last twenty years of layers were stacked on problems people were actually living with, yet those same years were also a history of deskilling, with every new layer breeding a generation that no longer needed to know what lay beneath it, and I don't think those two readings contradict each other. The stack is likely to survive agents, and I think the sturdier reasons are not the usual training data inertia but **review accountability** and **the accumulated troubleshooting record**. Still, just as SQL won almost completely without protecting the job title of the people who knew it, **the survival of a stack and the value of the people who know it move independently.** The job title will likely be absorbed into product engineer while frontend expertise condenses into a few habitats, and what remains a problem is that the ladder for growing that condensed layer back is hard to see.

## The Past: Why the Layers Piled Up

Frontend complexity is a long-running joke. The line about needing five build tools to show a single HTML page has held up for a decade. But peel the layers back one at a time, and it's hard to find one that piled up for no reason.

It started with documents. Until the mid-2000s, a web page was a thing the server rendered into HTML and the browser painted, and changing anything meant reloading the whole page. What shook that premise in front of the public was Gmail (2004) and Google Maps (2005). A screen that updated without reloading the page was a strange experience at the time, and once this approach got a name, Ajax (2005), the movement to make the web an app rather than a document was underway.

The problem was that there were no decent tools for building that app. Every browser's DOM API differed slightly, and there was the giant exception called IE6. Developers kept browser compatibility tables at their elbow and stacked up conditionals. jQuery (2006) covered that fragmentation with a single API and became the de facto standard. That's the first layer.

As apps built with jQuery grew, the next problem surfaced. With direct DOM manipulation, it was hard to trace where and how state changed, and bugs where data and screen drifted apart multiplied in proportion to code size. Backbone and AngularJS (2010) tried to tame this with structure, and React (2013) gave a different answer: declare the UI as a function of state, and leave DOM updates to the library. JSX, which puts HTML inside JavaScript, was met with something close to ridicule when it was unveiled, but in the end this declarative model won out in the competition. The second layer.

Problems were piling up on the language side too. We were now building apps of hundreds of thousands of lines in a language with no module system. CommonJS and AMD proliferated until ES2015 standardized modules and new syntax, but there was a lag before browsers caught up. Babel (transpiling) and webpack (bundling) filled that lag, and from then on the build step became a constant in frontend work. The origin of the five-build-tools joke, and the third layer.

Layers begat layers. Build tools written in JavaScript couldn't keep pace with growing codebases, so tools rewritten in Go and Rust (esbuild, SWC), and Vite on top of them, covered that problem. The fourth layer. SPAs that shipped everything to the client hit their limits on low-end devices and slow networks as bundles swelled to multiple megabytes, and the return to the server (Next.js's SSR and SSG, React Server Components, Astro's islands) covered that problem. The fifth layer, and the return to the starting point I mentioned at the top.

Every layer was a reaction to a problem the previous layer left behind. I spent twelve years in the middle of this current, and one thing I remember clearly is that each time a layer appeared, the problem it was trying to solve was real.

But the same history can be read from the other side. Every time a layer was added, a generation was born that didn't need to know what was under it. The jQuery generation memorized browser compatibility tables, but the generation after them didn't have to, and the framework generation could build screens without deeply knowing HTML semantics, HTTP caching, or accessibility. This is why people point out that as the barrier to entry dropped, the median quality of the output dropped with it, and why some call this period a lost decade.

The two readings look contradictory, but I think they are really the front and back of the same fact. The layers arose because they were needed, and once they existed, they made what was beneath them invisible. That is simply the nature of abstraction: it covers a problem to protect you from it, and at the same time makes you forget what it covered. This duality will come back once more when we talk about the future. (For an essay that walks down these layers one by one and defends them, see David Poblador's [The Descent](https://davidpoblador.com/deep-dives/what-happened-to-the-frontend/); for one that indicts the same period as a lost decade, see Mauro Bieg's [essay](https://mastrojs.github.io/blog/2026-05-23-is-AI-causing-a-repeat-of-frontends-lost-decade/).)

## The Present: The Return, and the Change of Author

As I said at the top, the pendulum has come back to the server. If we stopped here, this would just be a story about history moving in cycles. But at the exact moment of the return, the hand writing the code is changing. [As I quoted in an earlier essay](https://yceffort.kr/2026/06/learning-what-ai-cant-do), both Microsoft and Google say around 30% of their new code is written by AI. How each company counts is murky, but the direction seems clear.

Why does this become a problem for the stack? Because the layers of the last twenty years were mostly stacked on **human** problems. JSX is a syntax for letting a person see markup and logic at a glance, component boundaries are units of human cognition, and the rules of hooks (call only at the top level, never in conditionals) are guardrails built so people don't make mistakes. Lint, types, framework conventions: most of it was designed against human limitations. Once the author is no longer human, a question naturally arises. Is this accumulation an asset, or a cost?

From here on, this is a story about the future.

## Future 1: The Dismantling Scenario

It's only fair to engage the strongest form of the opposing argument, so let me lay out the case for dismantling as forcefully as I can.

Abstractions tuned to human cognitive limits leave nothing but cost for an author with no cognitive limits. If agents both write and read the code, there is no reason a vendor wouldn't design and push a new agent-friendly target (something more deterministic, easier to verify, expressible in fewer tokens) instead of human DX. And model vendors have plenty of incentive to do exactly that. It's hard to find a moat sturdier than having the target your own model handles best become the industry standard.

The standard rebuttal here is training data inertia. Most of the world's frontend code is written in React and its ecosystem, the models were trained on that data, so agents will keep producing that stack. There's something to this, but it's less sturdy than it looks. Inertia compounds, but it is vulnerable to exogenous shocks. If a vendor builds a new target, trains its own models on it intensively, fills the early gaps with synthetic data, and makes it the default in its own agent products, the inertia could flip within a few years.

Up to this point, the story is entirely plausible. But there is one thing I want to point out. This scenario skips the last link in the causal chain.

## Future 2: Even If They Push, People Don't Take It

Follow the causal chain to the end, and the last link is not the model but adoption. And adoption is done by people. Even at the stage where agents write all the code, a person presses the deploy button, and a person gets dragged out of bed at dawn when there's an outage. That person sits in a seat that requires at least minimal verification. But verification is only possible on something you can read. Code written in an unfamiliar stack easily becomes a black box that's hard to review, and it's hard to sign off on code like that with your name on it. So in the end, people are likely to choose the familiar stack they can read.

I think it matters that this bottleneck is not a technical problem. In [an essay on reading code](https://yceffort.kr/2026/06/do-you-need-to-read-code) I wrote that accountability is not a matter of understanding in your head but a matter of contracts, and the same structure operates here. Review accountability is bound up in organizations and law, so model capability alone cannot dissolve it. However high agent autonomy climbs, as long as responsibility stays with people, the requirement that a person be able to read the code does not go away.

That said, it's more accurate to keep the scope narrow. Strictly speaking, this argument rests not on "familiarity" but on "reviewability," and there is a region where the two diverge: code that effectively never gets reviewed. Throwaway prototypes, marketing pages, build-and-discard MVPs. In those regions, as long as the result works, the stack hardly matters. So the precise proposition is something like "in regions where review accountability exists, the familiar stack has the advantage." Since code with revenue and maintenance on the line makes up the bulk, the overall outcome will tilt toward lock-in, but a hole is still a hole. It's also exactly the region I conceded at the end of that essay: "code that needs no understanding really exists, and that region is not small, and it is growing."

The dismantling scenario can drive through this hole. An agent-optimized target gains a foothold in the review-free region, matures there, and then climbs up into the reviewed region. It's a picture you can readily draw.

## Future 3: Even Code Nobody Reviews Still Breaks

And yet even in that hole, I think the existing stack has the edge. Because the information needed for troubleshooting is already piled up there.

Even a build-and-discard prototype has to be fixed if it stops working right before the demo. What you need at that moment is not review capability but a rope to grab. Agents still get stuck fairly often, and when they do, the problem lands back on a person. This is the field version of what Joel Spolsky called [the Law of Leaky Abstractions](https://www.joelonsoftware.com/2002/11/11/the-law-of-leaky-abstractions/). At exactly the point where the abstraction leaks, the volume of information accumulated per stack becomes decisive. React and its ecosystem carry well over a decade of error messages, GitHub issues, and Stack Overflow answers. With an unfamiliar stack, even if humans have given up on review, there is nothing much to grab when it breaks.

One more point matters here: the agent itself was trained on that same rope. Even an agent's ability to correct its own errors improves with the amount of debugging material for that stack in its training data. In other words, training data inertia operates twice, once at generation and once again at self-correction. The common claim that "models write React well" is half the story; the other half is that "models fix React well." A new target might imitate the former with synthetic data, but the latter is an accumulation of records of things breaking in the field, which is hard to imitate for now.

The reason this argument is harder to route around than the review argument is simple. Review can be skipped depending on the situation. Code that doesn't work cannot be skipped.

To sum up, the forces supporting lock-in come in three layers: training data inertia (generation), review accountability (before the fact), and the troubleshooting record (after the fact). The first is a conditional argument that exogenous shocks can shake, and the second is a conditional argument that splits by region, but the third covers even the regions the other two exempt. That is the case for lock-in.

## The Shelf Life of These Arguments

If I stopped here, I'd have told only one side of the story. It seems more honest to also write down where these arguments weaken.

First, the review accountability argument does not guarantee "reading the code" forever. There is a precedent: the compiler. People once read the assembly compilers produced, but as trust accumulated, review didn't disappear; it moved up a floor. Now we verify the source and don't look at the assembly. What review accountability actually demands is "a verification interface humans can trust," not code reading itself. Right now that interface is code, but if E2E tests, visual regression, and agent QA become sufficiently fine-grained, an equilibrium of "verify the behavior, don't read the code" becomes possible too. The verification layer I sketched in [the essay on reading code](https://yceffort.kr/2026/06/do-you-need-to-read-code) is close to a blueprint for that move. Frontend is a domain where the correct answer is hard to specify (visuals, UX), so the transition will come late here, but it isn't blocked in principle.

CSS is the case that shows this "late" is not a vague hedge. It's something everyone who has used agents runs into: a model that handles logic just fine flounders conspicuously at layout and styling. Pick apart the reasons and it's no accident. First, the correct answer in CSS lives not in text but in the rendered screen. Workflows that take screenshots and feed them back have emerged, but today's visual judgment is low-resolution. It catches an obviously broken layout, but it can't quite adjudicate a few pixels of misalignment or the subtle rightness of an overlap. Next, a single rule acts globally. This is a language where one line of `position: relative` changes the stacking context and alters overlapping across the whole screen, which is why [analyses conclude that you can't predict the result from a code snippet alone](https://dev.to/asafaeirad/why-css-is-so-hard-for-generative-ais-to-understand-17fo). Finally, and this is the reason I find most fundamental, CSS doesn't raise an error when it's wrong. Compilation doesn't fail, no exception is thrown; something just looks off somewhere. An agent's self-correction loop needs a failure signal to run, and CSS doesn't give one. I wrote earlier that "code that doesn't work cannot be skipped," and CSS is precisely the blind spot of that proposition. It works even while wrong. The point about troubleshooting information piling up on the internet is also weak for CSS. The questions and answers are vast, but the solutions are bound to each asker's context and don't transfer well to a different screen.

What's interesting is the relationship between this weakness and tool choice. There's a claim that the utility-class approach, which gathers structure and style in one place (Tailwind), is favorable for models to handle. Though it's easy to overstate: Tailwind had already established itself before agents, for human reasons (the locality of styles sitting next to markup), and its machine-friendliness is more of a property discovered after the fact. Still, I think the direction is telling. If code shaped for machine predictability gains an advantage under selection pressure, then the "reorganization toward agent-friendly targets" from Future 1 is likely to arrive not as a revolutionary replacement but as this kind of quiet drift. So CSS ends up as evidence for both sides: for lock-in, in that the last few pixels of the screen will likely remain a human's job until visual verification matures; for dismantling, in that tool choice has already begun tilting toward the machine.

The second weakness is on the troubleshooting side. This moat has a structure that eats itself. The moat is a dozen-plus years deep because humans write slowly. Once agents start writing a new stack, debugging cases and telemetry accumulate at machine speed. It may be a two-to-three-year moat rather than a fifteen-year one. Of course, the problem of having no initial foothold remains, so the honest assessment is probably that it holds for now but is not permanent.

Finally, there is one path left by which all three layers collapse at once: the moment agent debugging shifts from information retrieval to reasoning. If agents reach the level of unwinding errors from first principles without a pile of cases for a particular stack, all three arguments lose their footing. Whether that moment will come, I don't know, and the signals that it will still look weak to me. Until then, today's stack will remain.

## Future 4: Winning, Then Turning Transparent

Up to here the question was "which stack wins," and my answer was lock-in. But this question itself may not be the crux.

The case of SQL suggests as much. SQL won just about completely. It hasn't been replaced in fifty years, and all three of the arguments I just built hold for it: the most training data, human reviewability, the largest troubleshooting record. And yet the job title "SQL developer" has all but disappeared. The work of handling SQL didn't disappear. That work was absorbed into the data disciplines, everyday SQL came to be written by ORMs and generation tools, and SQL knowledge became common sense rather than a premium. Which is to say, it stopped being a profession on its own. On the opposite side stands COBOL. Equally locked in, but as the supply of people who knew it dried up, it became a market where retired developers get called back at a premium, a story that resurfaces periodically. From the same outcome, lock-in, the fortunes of the people who knew each stack split in opposite directions.

Look at the three arguments again and they all defend "which stack the output comes out in." None of the three answers "does knowledge of that stack become a person's price?" Even the review accountability argument is like that. The moment review moves from code reading to behavior verification, the stack becomes a layer that wins and that nobody reads. Written by machines and fixed by machines, a transparent substrate, so to speak. In that world, "I know React" will carry about as much distinction as "I know SQL" does today.

The fork is between the COBOL type and the SQL type, and I think the SQL type is more likely. COBOL protected the value of its people because it became a language in which no new code is written, so the supply of people who knew it dried up. Today's frontend stack is the opposite: a living substrate onto which agents pour new code every day, and knowledge of a living substrate does not become scarce.

And at this point, the duality from the section on the past returns. I said abstraction covers a problem and makes you forget what it covered. Just as frameworks pushed the web's fundamentals below the waterline of common sense, agents push framework knowledge itself below it. The descent doesn't stop; it goes down one more floor. At every layer, the experts of that layer believed this layer was different, and every time, the next layer came.

## The Number 90%

After writing down the path by which the three arguments collapse, there is a sentence one is tempted to write: that by the time debugging shifts to reasoning, 90% of frontend developers will be gone. Before writing that sentence as-is, we need to distinguish what this 90% is counting.

What disappears is coding labor. As I wrote in [the essay on job boundaries](https://yceffort.kr/2026/06/when-job-titles-blur), judgment, specification, and ownership remain even when code production becomes free, and in fact grow more expensive. The role of deciding what to build, conveying to the agent precisely "what must not happen," and responding at dawn when there's an outage. Since a large share of today's job really is typing code, "90% of coding labor" is a defensible number. But read it as "90% of the job," and you repeat the common equation that treats the extinction of writing-and-fixing labor and the extinction of frontend developers as the same event. They are not the same event.

But this optimism has a precedent of failure: type design. Even in a discipline where judgment and taste are the core, [there's the observation that designing new typefaces is no longer a sustainable full-time profession](https://mastrojs.github.io/blog/2026-05-23-is-AI-causing-a-repeat-of-frontends-lost-decade/). "Judgment doesn't disappear" does not guarantee "the seats where judgment pays the bills don't shrink." Judgment can remain while the seats that sell it for money narrow.

There's a variable pointing the other way, too. The 90% is a number premised on fixed demand. Historically, when production costs collapse, demand expands. Websites became apps, and apps became the UI of everything. If Jevons's paradox, where coal consumption rose when coal efficiency improved, repeats in software, then even if labor per unit falls 90%, total volume growing tenfold means headcount shrinks far less. Of course, there's also a world where the expansion simply doesn't happen. If it's not a world with more UI but a world where UI itself is replaced by conversation with agents, there's no vessel for the expansion in the first place.

So the honest answer on 90% is probably this. If it points at coding labor, probably yes. If it points at the whole job, possible but undetermined. One thing is comparatively clear: the remaining 10% will not be the same work as today's 10%.

Sketching that 10% a bit more concretely: the job title will converge on product engineer. Once code production becomes common sense, the modifier "frontend" will fall off the title the way it did for SQL, and the likely shape is a person who holds judgment, specification, and ownership taking responsibility for the screen along with everything else. Frontend expertise itself, though, will condense rather than disappear. Even today, the people who truly understand the rendering pipeline or the internals of bundlers are a tiny few, and that tiny few cluster at the vendors who build the stack and the platform teams of massive services. After agents, this condensation goes one floor further, and I'd guess three habitats remain: the vendors who build the stack, the massive services where edge cases are daily life, and an on-call specialist market (today's web performance consultants and accessibility auditors are its prototype). What they do won't be only troubleshooting, either. It will be closer to designing the platforms and verification gates that keep hundreds of agents from breaking the screen. A path similar to how SRE started as a discipline that caught outages and settled into a small elite that designs platforms.

There is one unstable corner in this picture, though. The ability to catch edge cases comes from the experience of catching ordinary cases in bulk, and the track absorbed into product engineering doesn't accumulate that experience. In [the essay on why judgment stops being cultivated](https://yceffort.kr/2026/06/learning-what-ai-cant-do) I wrote that work no longer trains people for free, and this is the job-track version of that problem. The condensed expert layer is left with its reproduction ladder cut, and when the current generation steps back, that seat may become a COBOL-style premium market. Remaining as a tiny few and remaining stably maintained are different problems.

To sum up, the remaining 10% will be a job reconstituted around judgment, specification, and ownership, with fewer people and a higher unit price for those who remain. And whether that seat stays a "profession" or becomes a small niche the way type design did will, in the end, be decided by the direction of demand.

## The Stack's Victory and My Survival

Looking back, the layers of the past piled up because they were needed, and once piled, they made it unnecessary to know the expertise beneath them. Evolution and deskilling are the front and back of the same history. We are now at the moment when, having just returned to server rendering and minimal JavaScript, the hand writing the code is shifting from human to agent. I think the stack is likely to remain, less because of training data inertia than because of review accountability and the troubleshooting record. Review can be skipped, but code that doesn't work cannot, and the only path by which this structure collapses is debugging shifting from retrieval to reasoning, a signal that still looks weak.

But the stack's victory does not guarantee a person's seat. The SQL path, winning and then turning transparent, looks more likely: the job title gets absorbed into product engineer, and expertise condenses into three habitats, namely vendors, platform teams, and the on-call market. The problem that the ladder for growing that layer back is cut remains as well. So what to prepare for, I think, is less knowing the stack better than [designing and owning the verification](https://yceffort.kr/2026/06/do-you-need-to-read-code) when the verification interface moves from code to behavior. What I wanted to say in this essay comes down to this: the stack's victory and my own survival are separate questions.

> Related essays: [If the Code Meets Spec and the Bugs Get Fixed, Does It Matter That You Can't Read It?](https://yceffort.kr/2026/06/do-you-need-to-read-code), [If AI Erases the Boundaries Between Planning, Development, and Design, What Remains?](https://yceffort.kr/2026/06/when-job-titles-blur), [When Was the Last Time You Read Code Seriously?](https://yceffort.kr/2026/06/learning-what-ai-cant-do). They share the concerns of the "Judgment in the AI Era" series.

## References

- [The Descent: What Happened to the Frontend While You Weren't Watching (David Poblador i Garcia, 2026)](https://davidpoblador.com/deep-dives/what-happened-to-the-frontend/)
- [Is AI causing a repeat of Frontend's Lost Decade? (Mauro Bieg, 2026)](https://mastrojs.github.io/blog/2026-05-23-is-AI-causing-a-repeat-of-frontends-lost-decade/)
- [The Law of Leaky Abstractions (Joel Spolsky, 2002)](https://www.joelonsoftware.com/2002/11/11/the-law-of-leaky-abstractions/)
- [Why CSS Is So Hard for Generative AIs to Understand? (ASafaeirad, 2025)](https://dev.to/asafaeirad/why-css-is-so-hard-for-generative-ais-to-understand-17fo)
