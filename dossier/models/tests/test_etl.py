'''dossier.models.tests for ETL

.. This software is released under an MIT/X11 open source license.
   Copyright 2015 Diffeo, Inc.
'''


from dossier.models.tests import nltk_data
from dossier.models.etl import html_to_fc


test_html = '''

    <body id="ViewAd">
  <!-- Google Tag Manager -->
  <noscript><iframe src="//www.googletagmanager.com/ns.html?id=GTM-5KCSP8"
  height="0" width="0" style="display:none;visibility:hidden"></iframe></noscript>
  <script>(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start': 
  new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],
  j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src=
  '//www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);
  })(window,document,'script','bpDataLayer','GTM-5KCSP8');</script>
  <!-- End Google Tag Manager -->
    <div id="tlHeader">
<div id="postAdButton">
<form name="formPost" id="formPost" action="http://posting.newyork.backpage.com/online/classifieds/PostAdPPI.html/nyc/newyork.backpage.com/" method="get">
      <input type="submit" value="Post Ad" class="button" id="postAdButton">
      <input type="hidden" name="u" value="nyc">
      <input type="hidden" name="serverName" value="newyork.backpage.com">
    </form>
        </div><!-- #postAdButton -->
          <span class="formSearchHideOnSmallScreens">
          <input type="text" size="15" name="keyword" value=" keyword" onFocus="if (document.formSearch.keyword.value == ' keyword') document.formSearch.keyword.value = ''; return true;" maxlength="100">
            <select name="section">
                  <option value="26197783">local places
                  <option value="4382">community
                  <option value="4378">buy/ sell/ trade
                  <option value="153676">automotive
                  <option value="4380">musician
                  <option value="4376">rentals
                  <option value="4375">real estate
                  <option value="4373">jobs
</select>
Traveling across the country to buy real estate.
</body>
'''

def test_html_to_fc(nltk_data):
    
    fc = html_to_fc(test_html.decode('utf8'))
    assert 'Date' not in fc['meta_clean_html']
    assert 'bowNP_unnorm' in fc
    assert set(fc[u'bowNP_unnorm'].keys()) == set(['real estate', 'country'])
