---
layout: archive
permalink: /machine-learning/
title: "Machine learning Posts by Tags"
author_profile: true
header:
    image: "/images/dog.jpg"
---

{% include base_path %}
{% include group-by-array collection=site.posts field="tags" %}

{% for tag in group_names %}
  {% assign posts = group_ ---items[forloop.index0] %}
  <h2 id="{{ tag | slugify }}" class="archive__subtitle">{{ tag }}</h2>
  {% for post in posts %}
    {% include archive-single.html %}
  {% endfor %}
{% endfor %}