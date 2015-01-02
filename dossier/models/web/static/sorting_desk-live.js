
/* ------------------------------------------------------------
 * Load modules and respective dependencies.                    */

/* TODO: break up long line below.
 * This is a temporary measure to ensure the function block is indented as
 * intended. */
require(["SortingDesk", "SortingQueue", "API-SortingDesk", "DossierJS"], function (SortingDesk, SortingQueue, Api, DossierJS) {

  var loading = $("#loading"),
      nitems = $("#items"),
      nbins = $("#bins"),
      windowHeight = $(window).height(),
      requests = 0;

  $(".wrapper").fadeIn();
  nitems.height(windowHeight / 3);
  nbins.height(windowHeight - 40); /* 40 = vertical padding and margin estimate */

  /* ------------------------------------------------------------
   * Specialise SortingDesk classes.
   * --
   * ControllerBinSpawner <-- ProtarchBinSpawner */
  var ProtarchBinSpawner = function (owner, fnRender, fnAdd)
  {
    SortingDesk.ControllerBinSpawner.call(this, owner, fnRender, fnAdd);
  };

  ProtarchBinSpawner.prototype =
    Object.create(SortingDesk.ControllerBinSpawner.prototype);

  ProtarchBinSpawner.prototype.initialise = function ()
  {
    var self = this;

    /* Install custom handler for instantiation of new bins. Specifically, a
     * bin is created when the user drops an item anywhere on the page. */
    new SortingQueue.Droppable($("body"), {
      classHover: this.owner_.owner.options.css.droppableHover,
      scopes: [ 'text-item' ],

      drop: function (e, id) {
        id = decodeURIComponent(id);

        self.add(id);

        var items = self.owner_.owner.sortingQueue.items;
        items.remove(items.getById(id));
      }
    } );
  };

  ProtarchBinSpawner.prototype.reset = function ()
  {
    /* Invoke base class method. */
    SortingDesk.ControllerBinSpawner.prototype.reset.call(this);
  };


  /* ------------------------------------------------------------
   * Initialise API and instantiate SortingDesk. */
  var sd = new SortingDesk.Sorter( {
    nodes: {
      items: nitems,
      bins: nbins,
      buttonDismiss: $("#button-dismiss")
    },
    constructors: {
      ControllerBinSpawner: ProtarchBinSpawner
    },
    visibleItems: 15
  }, $.extend(Api, {
    onRequestStart: function () { if(!requests++) loading.stop().fadeIn(); },
    onRequestStop: function () { if(!--requests) loading.stop().fadeOut(); }
  } ) );

  var api = new DossierJS.API(DOSSIER_STACK_API_URL);
  var $sel_engines = $('#search-engine select');
  api.searchEngines().done(function(search_engines) {
    search_engines.forEach(function(name) {
      var $opt = $('<option value="' + name + '">' + name + '</option>');
      if (Api.getSearchEngine() == name) {
        $opt.attr('selected', true);
      }
      $sel_engines.append($opt);
    });
  });
  $sel_engines.change(function() {
      Api.setSearchEngine(this.options[this.selectedIndex].value);
      sd.sortingQueue.items.removeAll();
  });
});

