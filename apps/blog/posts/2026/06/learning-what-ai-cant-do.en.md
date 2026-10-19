---
title: 'When Was the Last Time You <em>Read Code Seriously</em>?'
tags:
  - essay
  - ai
  - career
  - software-engineering
  - code-review
published: true
date: 2026-06-21 11:00:00
updated: 2026-10-19 13:52:52
description: 'The friction that made judgment expensive is the same friction that taught it. Why the skill growing most valuable in the AI era is the one being cultivated least'
series: 'Judgment in the AI Era'
seriesOrder: 3
art:
  undraw: open-book
  layout: glyph
  hue: violet
  tone: dark
  hero: 'desirable difficulties'
---

## The Signal Goes Dark

When was the last time you read code seriously? By "seriously" I don't mean running your eyes over the lines. I mean following it all the way through: why it was written this way, what could break here, whether what this code claims is actually true.

I started to write "this question isn't aimed at me," and then stopped. I still read. More precisely, I still try to read. But honestly, that reading has been getting hazy lately. There are days when I open a diff, see that the tests are green, and feel the concentration that was supposed to carry me to the end quietly slip away. Still, most of the time, I read. So this question is aimed, before me, at the people writing code these days, or more precisely, the people telling something else to write it. And yet, honestly, it bothers me that I'm sitting in a safe enough seat to pose this question to others. This is not an idle question, either. The _reasons_ to read code seriously are draining away on every side.

First, production has left human hands. Microsoft and Google both [say](https://www.itpro.com/software/development/developers-will-need-to-adapt-microsoft-ceo-satya-nadella-joins-googles-sundar-pichai-in-revealing-the-scale-of-ai-generated-code-at-the-tech-giants-and-its-a-stark-warning-for-software-developers) that around 30% of their new code is written by AI. How each company counted is murky, but the direction is unmistakable.

Next, evaluation has become a formality. Code review is converging on a procedure: open the diff, check that the tests are green, approve if nothing looks off at the big-picture level. Reading every line to the end happens only when something really, truly smells, and that frequency drops as AI gets better.

Finally, interviews are steering around code too. Sitting on the interviewer's side, what we asked candidates about wasn't data structures or code they'd written by hand. It was how they use AI, how they notice when an agent is wrong, how far they delegate and where they stop.

Taken separately, each of these three scenes is reasonable. But the place where code was _evaluated_ (review) and the place where coders were _selected_ (interviews) have, in the same period, as if by agreement, stopped looking at code deeply. That humans are stepping back from making code: fine, let's grant that. The problem is that the signals for _judging_ code are going dark along with it.

And of all things, the capability that's going dark (the judgment that knows what's correct and where to be suspicious) is exactly the capability that's growing more expensive. That's what I argued in my last two essays, on [reading code](https://yceffort.kr/2026/06/do-you-need-to-read-code) and on [the blurring of job boundaries](https://yceffort.kr/2026/06/when-job-titles-blur): making things becomes commonplace, while judgment and responsibility remain and grow dearer. This essay is the dark mirror of those two. The capability that's growing more expensive is, at the same time, being cultivated less and less. And this isn't a story about anyone getting lazy. It's a story about the need to learn disappearing.

## The Disappearance Is Not a Feeling

So far this is about production and evaluation leaving human hands. That part is easy to accept. The harder claim comes next: that in the process, human _ability_ and _learning_ are being eroded too. Impressions alone won't carry this. It's too easy to dismiss as "an old-timer whining in front of a new tool." And that suspicion is legitimate. Almost no one writes assembly by hand anymore, and software did not collapse. Plato said writing would ruin memory (Phaedrus), yet letters let us think farther. The complaint that a new tool erases an old skill has been made every time, and it has been wrong every time. The prior probability is stacked against this essay from the start.

Still, there is evidence that this isn't just a feeling. Not much, but some.

There's the METR [experiment](https://metr.org/blog/2025-07-10-early-2025-ai-experienced-os-dev-study/) from 2025, which randomly assigned tasks to experienced open-source developers and measured the results. Tasks done with AI were on average 19% _slower_ than tasks done without. Yet the same developers felt they had been 20% _faster_ thanks to AI. Perception and measurement are inverted wholesale. Why this inversion happens, we'll come back to in the next section. It touches the core of this essay.

A [study](https://arxiv.org/abs/2601.20245) by Shen and Tamkin from 2026 takes a different angle. They split developers learning a new async library by whether they had AI assistance. The AI-assisted group developed less conceptual understanding, less code-reading ability, and less debugging skill, while gaining, on average, no meaningful efficiency benefit either. Work happened, but learning did not.

The code itself bears traces too. In GitClear's [report](https://www.gitclear.com/ai_assistant_code_quality_2025_research) analyzing 211 million lines across 2020 to 2024, the proportion of copy-pasted lines rose from 8.3% to 12.3%, and the proportion of code moved through refactoring fell from 25% in 2021 to under 10% in 2024. Tearing down and reworking is shrinking; stamping out and duplicating is growing. Toward accumulating output, not accumulating understanding.

Let me draw a line here. The claim that AI erodes human ability can easily sound like a well-worn critique of capitalism: the gains from rising productivity go to capital rather than to the people doing the work, and jobs once performed whole by skilled craftsmen are chopped into simple tasks anyone can do. This deskilling argument was laid out by Braverman in _Labor and Monopoly Capital_ back in 1974, a story fifty years old by now. The proposition "technology erodes skill" is, in itself, nothing new. That general thesis is not what I'm after.

Nor will I generalize that "nobody learns anymore." That's false. In every era there has been the 1% who dig all the way down, and they exist now too. There have always been people who don't learn, too. What changed is not the extremes but the middle. Something that propped up the median has fallen out. Not the curiosity of the 1%, not the excuses of the lazy, but the thing that forced most people in the middle to learn even against their will. What that thing is, is the next section.

## Two Edges of One Blade

Let's first narrow down what judgment means. Nothing grand. Knowing whether this abstraction is right, where this code will break, what needs verifying and what deserves suspicion. The thing AI won't take responsibility for on your behalf. The work of _producing_ code has gone to agents, but the work of _judging_ whether the product is correct still remains with people. That was where my last essay ended up.

The problem is where that judgment came from. I didn't learn it from books. I learned it by building, breaking, and debugging myself. By watching a wrong abstraction I'd erected collapse a few weeks later, by staring at logs until dawn because I couldn't find one line, by living through the moment when something I'd waved past with "this should be good enough" blew up in production. The hours of being stuck, lost, and struggling _were_ the learning. What passed easily left nothing behind.

This isn't my sentimentality; it's a fact that learning science has nailed down. It's what Robert Bjork calls desirable difficulties. Conditions that make learning _slow and hard_ are, paradoxically, what produce learning that _lasts_. Information that slides by without resistance feels well understood in the moment but evaporates almost immediately. What you dredge up haltingly, with effort, is what stays. The struggle is not a cost. It's the mechanism.

And here the double edge comes into view.

The very properties that made judgment _expensive_ (friction, struggle, getting stuck, slowness) are exactly the properties that used to _cultivate_ judgment. These are not two different things but two edges of the same one. It was scarce because of friction, and it was cultivated because of friction. Being expensive and being learnable grow from the same root.

And what AI removes is exactly that friction.

AI removes the getting-stuck. It hands you the right abstraction before you can erect the wrong one; it fixes the bug before you spend a night digging through logs. As a tool, this is excellent; it removes cost. But that cost was the very mechanism of learning. The moment friction is removed, judgment becoming expensive and judgment going uncultivated happen simultaneously. Not two separate problems but two consequences of one cause. One side makes judgment scarce (expensive); the other makes it unlearnable (uncultivated). Both flow from removing the same friction.

Now back to the METR inversion. The mystery of developers who got 19% slower with AI yet felt 20% faster.

It's not a mystery. It's a predicted result. In the same line of Bjork's research there's something called the fluency illusion: when information is processed smoothly and effortlessly, we mistake that very ease for a signal that "I understand this." It's the confusion of familiarity with competence: reread a textbook enough times and it looks familiar, so you feel you know it, until you try to recall it and nothing comes. Code that AI produces is smooth. It reads fluidly, looks plausible, offers no resistance. We misread that fluency as understanding. And we misread the absence of effort as evidence that the work is going well. The stuck-ness is gone, so it feels faster, but that stuck-ness was precisely the signal telling me what I didn't know. With the signal dark, you don't even know that you don't know. A [study](https://dl.acm.org/doi/full/10.1145/3706598.3713778) of 319 knowledge workers by Microsoft and CMU captured this with measurement: the more people trusted AI, the less critical thinking they did; the more they trusted themselves, the more they did. Which is to say, we set down judgment in proportion to how much we believe the fluency. That is also why perception and measurement point in opposite directions. When friction disappears, the _feeling_ of moving faster arrives first, and ability quietly falls behind.

## Where the Need Went

Everything so far is about ability. Remove friction, and ability goes uncultivated. But the more frightening part is on the motivation side. It's not only the friction that built ability that has vanished; the _need_ that made people endure that friction has drained away with it. It drains through two channels.

First, passively. It used to be that when you got stuck, you had to learn. Being stuck was itself compulsion. If you couldn't understand this error you couldn't move on to the next thing, so you dug in whether you liked it or not. But AI melts the stuck-ness on the spot. With nothing to be stuck on, there's no trigger to learn. It's not that people refuse to learn; the _moment_ that demands learning never arrives. Being stuck was the entrance to learning, and that entrance has closed.

Second, actively. This channel is more vicious. Real-world work scores throughput: how fast, how much you cleared. On that yardstick, learning is not merely unnecessary; it's a _liability_. The person who digs into a blocker themselves instead of handing it to AI is, that quarter, the slow person. Slow loses on the scorecard. Surplus capacity gets absorbed into doing more work with fewer people ([the layoffs were covered in my last essay](https://yceffort.kr/2026/06/when-job-titles-blur)), and per-person slack (the margin that doesn't show up in immediate output) gets squeezed out. The very idle time that learning requires is the first thing cut.

Organizations talk quality with their mouths. What they actually reward is speed. Whatever the documents say, watch who gets praised at quarter's end and who slides down the review rankings, and the real yardstick reveals itself. Not what is said but what is rewarded is the real rule. And the rule that gets rewarded is speed.

The signals on the individual side point the same way. In Stack Overflow's [2025 survey](https://survey.stackoverflow.co/2025/ai/), the share of developers who believe AI output is accurate fell from 40% in 2024 to 29%, and those actively distrusting its accuracy (46%) outnumber those trusting it (33%). And yet 84% use it. Using a tool you don't trust isn't laziness. It's that not using it makes you the slow person. In a structure where avoidance is rational, depending on a tool you don't trust becomes just as rational.

So the collapse of motivation is not a moral failure. It is predicted behavior. Bjork showed that even when learners are told which methods work, absent compulsion, they drift back toward whatever feels easier and more fluent. Knowing does not change behavior. The crux of desirable difficulty is that it is _desirable_ but no one chooses it voluntarily. So the question "why doesn't anyone learn deeply anymore" is the wrong question. In an environment where both need and reward have turned their backs on learning, avoidance is not laziness but rational adaptation.

And this cannot be solved by willpower. When learning is purely discretionary and unrewarded, even the motivated few lose. If the desk next to mine is clearing work twice as fast with AI while I alone hug my blockers and learn slowly, I am simply the person who slides down the rankings. It's a collective action problem. In a structure where acting rightly alone means losing, even the will to act rightly becomes a cost.

And it isn't just the tool called AI that removes friction. The organization demanding throughput removes it too. The tool melts the blockers; the organization removes the slack to endure them. With friction disappearing from both sides, judgment grows expensive and goes uncultivated at once, and the need and motivation to learn it drain away with it.

## The Hand That Wants to Write a Prescription

I know how this essay is supposed to close. The obvious move is right there. "So manufacture friction deliberately. Set aside time to debug with AI turned off, don't ask the moment you get stuck…" My hand moves in that direction. And most essays these days are flowing exactly that way.

But that was precisely the ending of my last two essays. And it is advice that loses at equilibrium. If you take seriously what I just said (that when learning is discretionary and unrewarded, even the motivated few lose), then the prescription "individuals should deliberately manufacture friction" is nothing more than offloading a collective action problem onto individual willpower. Prescribing personal virtue for a problem the structure created sounds good and cannot beat the equilibrium. As long as the next desk is twice as fast, the person who hugs the friction alone is right, and loses.

What's more, this very discomfort (the itch to write a prescription) is the spot this essay is pointing at. Being able to issue a prescription means my judgment has already been cultivated. The costs are all paid, and now I'd be preaching the value of friction from the far side.

So there is no prescription. Instead, let me write down where this argument bends.

The strongest counterexample is the structure that owns outcomes: the mission-driven organization. Where you're evaluated on results rather than speed, and you operate what you build ("you build it, you run it," [the Vogels line quoted in part two](https://yceffort.kr/2026/06/when-job-titles-blur)), the active blade from the previous section dulls. The speed yardstick weakens. Moreover, owning outcomes brings some downstream friction back: taking your own on-call, putting out your own production fires at dawn. That is exactly the friction I described earlier ("I learned by living through things blowing up in production") and it is real learning. So this structure _partially_ weakens my thesis. Genuinely.

But follow it all the way and you end up back where you started. Being accountable for outcomes is a demand to _use_ judgment well, not a compulsion to _cultivate_ it. If the results are good, AI can write all the code; when the deadline is tight, you lean on it harder. So the friction of writing code yourself and colliding with it (the friction that cultivated judgment) does not come back. What comes back is only the friction of cleaning up incidents, and that is late and blunt. When an incident blows up you may learn "that choice was wrong," but it doesn't teach you why each line of code was dangerous. And if you hand even the cleanup to AI, that learning disappears too.

To sum up: evaluating by outcomes demands more judgment, while the friction that would cultivate judgment does not return. The demand rises; the supply stays flat. That is exactly the double edge. And even the signboard that says "we evaluate by outcomes" tends, in practice, to get swapped for speed metrics like deploy counts. It is harshest on the people just arriving. They are asked to produce, as outcomes, a judgment they do not yet possess, while the opportunity to cultivate it still isn't there. Seniors who already hold judgment draw on it and hold up fine; only the juniors stand exposed. Far from overturning my claim, this scene displays it exactly.

That much only sharpens the argument. The cases where I'd have to admit I'm wrong wholesale are separate. If the people entering the field right now, in their collisions with AI (in that new kind of wrestling where you notice when the agent is wrong and decide how far to delegate) are cultivating judgment through a different path than the old debugging, then I am wrong. Or if the disappearance of need is an illusion, and the friction has merely changed shape and remains, then I am also wrong. If either is true, this essay becomes an old-timer's whining. And as I conceded earlier, complaints of this kind have mostly been wrong. The odds are not on my side.

But from where I sit, that is not what I see. I see the opposite. The people who must pay the cost are the ones just entering, and they have neither the need nor the reward to endure the friction. The people who can see the problem most clearly are not the ones it falls on. That is why it doesn't get fixed. The people seated where this problem is most visible (myself included) are seated where they can say "not my problem." That asymmetry is the keystone.

I still read code (even if it gets hazy sometimes). But that's not because the need to read remains; it's because the judgment was already cultivated. The question of when you last read code seriously is, for the people entering now, steadily becoming "why would I?" And that _need_ is not coming tomorrow, either.

## References

- [Measuring the Impact of Early-2025 AI on Experienced Open-Source Developer Productivity (METR, 2025)](https://metr.org/blog/2025-07-10-early-2025-ai-experienced-os-dev-study/)
- [How AI Impacts Skill Formation (Shen & Tamkin, 2026)](https://arxiv.org/abs/2601.20245)
- [AI Copilot Code Quality: 2025 Research (GitClear)](https://www.gitclear.com/ai_assistant_code_quality_2025_research)
- [2025 Stack Overflow Developer Survey: AI](https://survey.stackoverflow.co/2025/ai/)
- [Satya Nadella: up to 30% of Microsoft code is written by AI (CNBC, 2025)](https://www.cnbc.com/2025/04/29/satya-nadella-says-as-much-as-30percent-of-microsoft-code-is-written-by-ai.html)
- [The Impact of Generative AI on Critical Thinking (Lee et al., Microsoft Research & CMU, CHI 2025)](https://dl.acm.org/doi/full/10.1145/3706598.3713778)
- Harry Braverman, [Labor and Monopoly Capital: The Degradation of Work in the Twentieth Century](https://monthlyreview.org/9780853459408/) (1974)
- Elizabeth L. Bjork & Robert A. Bjork, ["Making Things Hard on Yourself, but in a Good Way: Creating Desirable Difficulties to Enhance Learning"](https://bjorklab.psych.ucla.edu/wp-content/uploads/sites/13/2016/04/EBjork_RBjork_2011.pdf) (Psychology and the Real World, 2011)
