---
layout: post
title:  "My first post"
date:   2021-06-25
categories: post
---

<h2>{{ post.title }}</h2>

I'm learning how to make a site in Jekyll

This was published on {{page.date}}

{% for level in (1..3) %}
	<h{{level}}>This is a h{{level}} tag!</h{{level}}>
{% endfor %}


{% assign names = "billy, bob, joel" | split: ', ' %}
<ul>
    {% for name in names %}
    <li>{{ name | capitalize }}</li>
    {% endfor %}
</ul>