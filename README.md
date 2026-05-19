# Using the Pytorch profiler

![Pytorch Trace](./images/intro_trace.png)

Pytorch comes with a built in profiling tool that can produce traces which help
with diagnosing runtime performance issues and allow you to reduce the amount
of time it takes to train your models. The Pytorch document provides a great
tutorial on how to set up the profiler and produce a trace, but unfortunately
it doesn't really tell you how to use this trace once you have produced it.
There are a few other tutorials out there, including videos on youtube, but
in my opinion, I think they tend to jump straight to diagnosing and fixing
a problem and don't really explain what the trace file is showing and how to
read it.

