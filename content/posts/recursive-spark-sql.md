+++
title = 'Recursive Spark SQL'
categories = ['Development']
date =  2022-07-31T11:13:56-04:00
draft = true
+++

Spark is a general distributed computing framework designed on Hadoop YARN / Hadoop clusters that allows us to run computations that exceed the capacity of a single machine. 

Spark SQL is a subsystem of Spark....

However, Spark SQL does not support recursive SQL through SQL strings or the other syntax...

But we can use the imperative/iterative nature of Spark to reproduce this behaviour...
